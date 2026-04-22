"""Parameterized fused_norm_scale kernel for arbitrary M×N.

Computes out[i,j] = matrix[i,j] * rsqrt(variance[i,j]).

op_a: rsqrt(variance) = 1 / sqrt(variance)  — vsqrt.bf16 + vrecip.bf16
op_b: elementwise_mul(matrix, rsqrt_v)       — vmul.bf16

Both inputs and output stored in tiled row-major bf16 layout (2048 B per tile).

Constraints:
    - M and N must be multiples of 32.
    - Variance values must be positive.

VMEM slots:
    VMEM_VAR = 0x2000   2 KB — variance tile
    VMEM_MAT = 0x2800   2 KB — matrix tile
    VMEM_OUT = 0x3000   2 KB — output tile
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

VMEM_VAR = 0x2000
VMEM_MAT = 0x2800
VMEM_OUT = 0x3000

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


def make_fused_norm_scale_instructions(
    M: int,
    N: int,
    dram_var: int,
    dram_mat: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate the full instruction list for an M×N fused_norm_scale.

    MRF layout per tile:
        (v0, v1) = variance halves
        (v2, v3) = matrix halves
        (v4, v5) = sqrt(variance)
        (v6, v7) = rsqrt(variance) = 1/sqrt
        (v8, v9) = output = matrix * rsqrt
    """
    assert M % TILE == 0 and N % TILE == 0
    M_tiles = M // TILE
    N_tiles = N // TILE

    insns: list[Instruction] = []
    _emit_load_vmem_addr(1, VMEM_VAR, insns)
    _emit_load_vmem_addr(2, VMEM_MAT, insns)
    _emit_load_vmem_addr(3, VMEM_OUT, insns)
    _emit_load_imm32(4, TILE_BYTES_BF16, insns)

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    for m in range(M_tiles):
        for n in range(N_tiles):
            var_addr = dram_var + _bf16_tile_offset(m, n, N)
            mat_addr = dram_mat + _bf16_tile_offset(m, n, N)
            out_addr = dram_out + _bf16_tile_offset(m, n, N)

            _emit_load_imm32(5, var_addr, insns)
            _emit_load_imm32(6, mat_addr, insns)

            # DMA variance and matrix in parallel
            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=5, rs2=4, channel=0)))
            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=6, rs2=4, channel=1)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

            # Load variance pair (v0, v1)
            insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            # Load matrix pair (v2, v3)
            insns.append(Instruction("vload", VectorArgs(vd=2, rs1=2, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=3, rs1=2, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            # op_a: rsqrt = 1/sqrt(var)
            insns.append(Instruction("vsqrt.bf16", VectorArgs(vd=4, vs1=0)))   # (v4,v5) = sqrt(var)
            insns.append(Instruction("delay", ScalarArgs(imm=66)))
            insns.append(Instruction("vrecip.bf16", VectorArgs(vd=6, vs1=4)))  # (v6,v7) = rsqrt
            insns.append(Instruction("delay", ScalarArgs(imm=66)))

            # op_b: matrix * rsqrt
            insns.append(Instruction("vmul.bf16", VectorArgs(vd=8, vs1=2, vs2=6)))  # (v8,v9) = out
            insns.append(Instruction("delay", ScalarArgs(imm=66)))

            # Store
            insns.append(Instruction("vstore", VectorArgs(vd=8, rs1=3, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vstore", VectorArgs(vd=9, rs1=3, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            _emit_load_imm32(7, out_addr, insns)
            insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=7, rs1=3, rs2=4, channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def fused_norm_scale_reference(
    variance: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """output[i,j] = matrix[i,j] * rsqrt(variance[i,j]).

    Matches the two-step kernel: bf16 sqrt, bf16 recip, bf16 mul.
    """
    sqrt_v = torch.sqrt(variance.float()).to(torch.bfloat16)
    rsqrt_v = (1.0 / sqrt_v.float()).to(torch.bfloat16)
    return (matrix.float() * rsqrt_v.float()).to(matrix.dtype)


def _make_program(M: int, N: int, seed: int):
    dram_var = 0x0000
    dram_mat = dram_var + M * N * BF16_BYTES
    dram_out = dram_mat + M * N * BF16_BYTES

    torch.manual_seed(seed)
    variance = (torch.randn(M, N).abs() + 0.1).to(torch.bfloat16)
    matrix = torch.randn(M, N, dtype=torch.bfloat16)
    expected = fused_norm_scale_reference(variance, matrix)

    insns = make_fused_norm_scale_instructions(
        M=M, N=N, dram_var=dram_var, dram_mat=dram_mat, dram_out=dram_out
    )
    regions = [
        (dram_var, _tile_matrix_bf16(variance, M, N)),
        (dram_mat, _tile_matrix_bf16(matrix, M, N)),
    ]
    golden = (dram_out, _tile_matrix_bf16(expected, M, N))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, 32, seed=80)


class ParameterizedFusedNormScale32x32Program(Program):
    """fused_norm_scale on a single 32×32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, 64, seed=81)


class ParameterizedFusedNormScale64x64Program(Program):
    """fused_norm_scale on a 64×64 bf16 tensor (2×2 tiles)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_insns, _64x32_regions, _64x32_golden = _make_program(64, 32, seed=82)


class ParameterizedFusedNormScale64x32Program(Program):
    """fused_norm_scale on a 64×32 bf16 tensor (2×1 tiles)."""

    instructions: List[Instruction[Any]] = _64x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
