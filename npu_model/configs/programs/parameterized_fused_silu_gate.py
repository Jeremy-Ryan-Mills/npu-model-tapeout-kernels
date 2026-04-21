"""Parameterized fused_silu_gate kernel for arbitrary M×N.

Computes silu(x) = x * sigmoid(x) = x * 1/(1+exp(-x)).
Functionally equivalent to parameterized_silu but named after the
fused op (op_a: sigmoid, op_b: element-wise multiply).

Constraints:
    - M and N must be multiples of 32.

VMEM slots:
    VMEM_X   = 0x2000   2 KB
    VMEM_OUT = 0x2800   2 KB
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

VMEM_X = 0x2000
VMEM_OUT = 0x2800

TILE = 32
BF16_BYTES = 2
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES


def _emit_load_imm32(rd: int, value: int, out: list[Instruction]) -> None:
    if value == 0:
        out.append(Instruction("addi", ScalarArgs(rd=rd, rs1=0, imm=0)))
        return
    upper = (value + 0x800) >> 12
    lower = value - (upper << 12)
    if upper:
        out.append(Instruction("lui", ScalarArgs(rd=rd, imm=upper)))
        if lower:
            out.append(Instruction("addi", ScalarArgs(rd=rd, rs1=rd, imm=lower)))
    else:
        out.append(Instruction("addi", ScalarArgs(rd=rd, rs1=0, imm=lower)))


def _emit_load_vmem_addr(rd: int, vmem_addr: int, out: list[Instruction]) -> None:
    _emit_load_imm32(rd, vmem_addr, out)


def _bf16_tile_offset(m: int, n: int, N: int) -> int:
    return (m * (N // TILE) + n) * TILE_BYTES_BF16


def _tile_matrix_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for r in range(M_tiles):
        for c in range(N_tiles):
            tile = mat[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE].contiguous()
            parts.append(tile.reshape(-1))
    return torch.cat(parts)


def make_fused_silu_gate_instructions(
    M: int,
    N: int,
    dram_x: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate the full instruction list for an M×N fused_silu_gate.

    op_a: sigmoid(x) = 1/(1+exp(-x))
    op_b: silu(x) = sigmoid(x) * x
    """
    assert M % TILE == 0 and N % TILE == 0
    M_tiles = M // TILE
    N_tiles = N // TILE

    insns: list[Instruction] = []
    _emit_load_vmem_addr(1, VMEM_X, insns)
    _emit_load_vmem_addr(2, VMEM_OUT, insns)
    _emit_load_imm32(3, TILE_BYTES_BF16, insns)

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    for m in range(M_tiles):
        for n in range(N_tiles):
            x_addr = dram_x + _bf16_tile_offset(m, n, N)
            out_addr = dram_out + _bf16_tile_offset(m, n, N)

            _emit_load_imm32(4, x_addr, insns)
            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=4, rs2=3, channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

            # Load x pair
            insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            # Load constants
            insns.append(Instruction("vli.all", VectorArgs(vd=2, imm=-1)))  # -1.0
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vli.all", VectorArgs(vd=3, imm=-1)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vli.all", VectorArgs(vd=4, imm=1)))   # +1.0
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vli.all", VectorArgs(vd=5, imm=1)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            # op_a: sigmoid(x)
            insns.append(Instruction("vmul.bf16", VectorArgs(vd=6, vs1=0, vs2=2)))   # (v6,v7) = -x
            insns.append(Instruction("delay", ScalarArgs(imm=4)))
            insns.append(Instruction("vexp.bf16", VectorArgs(vd=8, vs1=6)))           # (v8,v9) = exp(-x)
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vadd.bf16", VectorArgs(vd=10, vs1=8, vs2=4)))  # (v10,v11) = 1+exp(-x)
            insns.append(Instruction("delay", ScalarArgs(imm=4)))
            insns.append(Instruction("vrecip.bf16", VectorArgs(vd=12, vs1=10)))       # (v12,v13) = sigmoid(x)
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            # op_b: silu(x) = sigmoid(x) * x
            insns.append(Instruction("vmul.bf16", VectorArgs(vd=14, vs1=12, vs2=0))) # (v14,v15) = silu(x)
            insns.append(Instruction("delay", ScalarArgs(imm=4)))

            # Store
            insns.append(Instruction("vstore", VectorArgs(vd=14, rs1=2, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vstore", VectorArgs(vd=15, rs1=2, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            _emit_load_imm32(5, out_addr, insns)
            insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=5, rs1=2, rs2=3, channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def fused_silu_gate_reference(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.sigmoid(xf)).to(x.dtype)


def _make_program(M: int, N: int, seed: int):
    dram_x = 0x0000
    dram_out = dram_x + M * N * BF16_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, N, dtype=torch.bfloat16)
    expected = fused_silu_gate_reference(x)

    insns = make_fused_silu_gate_instructions(M=M, N=N, dram_x=dram_x, dram_out=dram_out)
    regions = [(dram_x, _tile_matrix_bf16(x, M, N))]
    golden = (dram_out, _tile_matrix_bf16(expected, M, N))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, 32, seed=60)


class ParameterizedFusedSiluGate32x32Program(Program):
    """fused_silu_gate on a single 32×32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, 64, seed=61)


class ParameterizedFusedSiluGate64x64Program(Program):
    """fused_silu_gate on a 64×64 bf16 tensor (2×2 tiles)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_32x64_insns, _32x64_regions, _32x64_golden = _make_program(32, 64, seed=62)


class ParameterizedFusedSiluGate32x64Program(Program):
    """fused_silu_gate on a 32×64 bf16 tensor (1×2 tiles)."""

    instructions: List[Instruction[Any]] = _32x64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32x64_regions
    golden_result: tuple[int, torch.Tensor] = _32x64_golden
