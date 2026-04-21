"""Parameterized elementwise subtract kernel: C = A - B for arbitrary M×N.

Identical structure to parameterized_elementwise_add, using vsub.bf16.

Constraints:
    - M and N must be multiples of 32.

VMEM slots: VMEM_A=0x2000, VMEM_B=0x2800, VMEM_C=0x3000 (each 2 KB).
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

VMEM_A = 0x2000
VMEM_B = 0x2800
VMEM_C = 0x3000

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


def make_elementwise_sub_instructions(
    M: int,
    N: int,
    dram_a: int,
    dram_b: int,
    dram_c: int,
) -> list[Instruction]:
    """Generate the full instruction list for an M×N elementwise subtract."""
    assert M % TILE == 0 and N % TILE == 0
    M_tiles = M // TILE
    N_tiles = N // TILE

    insns: list[Instruction] = []
    _emit_load_vmem_addr(1, VMEM_A, insns)
    _emit_load_vmem_addr(2, VMEM_B, insns)
    _emit_load_vmem_addr(3, VMEM_C, insns)
    _emit_load_imm32(4, TILE_BYTES_BF16, insns)

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    for m in range(M_tiles):
        for n in range(N_tiles):
            a_addr = dram_a + _bf16_tile_offset(m, n, N)
            b_addr = dram_b + _bf16_tile_offset(m, n, N)
            c_addr = dram_c + _bf16_tile_offset(m, n, N)

            _emit_load_imm32(5, a_addr, insns)
            _emit_load_imm32(6, b_addr, insns)

            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=5, rs2=4, channel=0)))
            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=6, rs2=4, channel=1)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

            insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=2, rs1=2, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=3, rs1=2, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            insns.append(Instruction("vsub.bf16", VectorArgs(vd=4, vs1=0, vs2=2)))
            insns.append(Instruction("delay", ScalarArgs(imm=4)))

            insns.append(Instruction("vstore", VectorArgs(vd=4, rs1=3, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vstore", VectorArgs(vd=5, rs1=3, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=20)))

            _emit_load_imm32(7, c_addr, insns)
            insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=7, rs1=3, rs2=4, channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def elementwise_sub_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).to(a.dtype)


def _make_program(M: int, N: int, seed: int):
    dram_a = 0x0000
    dram_b = dram_a + M * N * BF16_BYTES
    dram_c = dram_b + M * N * BF16_BYTES

    torch.manual_seed(seed)
    a = torch.randn(M, N, dtype=torch.bfloat16)
    b = torch.randn(M, N, dtype=torch.bfloat16)
    expected = elementwise_sub_reference(a, b)

    insns = make_elementwise_sub_instructions(M=M, N=N, dram_a=dram_a, dram_b=dram_b, dram_c=dram_c)
    regions = [
        (dram_a, _tile_matrix_bf16(a, M, N)),
        (dram_b, _tile_matrix_bf16(b, M, N)),
    ]
    golden = (dram_c, _tile_matrix_bf16(expected, M, N))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, 32, seed=30)


class ParameterizedElementwiseSub32x32Program(Program):
    """Elementwise subtract on a single 32×32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, 64, seed=31)


class ParameterizedElementwiseSub64x64Program(Program):
    """Elementwise subtract on a 64×64 bf16 tensor (2×2 tiles)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_32x64_insns, _32x64_regions, _32x64_golden = _make_program(32, 64, seed=32)


class ParameterizedElementwiseSub32x64Program(Program):
    """Elementwise subtract on a 32×64 bf16 tensor (1×2 tiles)."""

    instructions: List[Instruction[Any]] = _32x64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32x64_regions
    golden_result: tuple[int, torch.Tensor] = _32x64_golden
