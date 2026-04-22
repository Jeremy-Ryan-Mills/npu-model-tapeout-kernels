"""Parameterized type-conversion kernel: bf16 ↔ f32 identity copy for M×N.

On this NPU, all VPU operations run in bf16.  A type conversion between
bf16 and f32 is an identity at the VPU level — the bf16 values are preserved
with no precision loss (bf16 is a subset of f32).

The kernel loads each 32×32 bf16 tile as an LMUL=2 pair (v0, v1), multiplies
by a pair of 1.0 constants (v2, v3) to produce the output pair (v4, v5), then
stores the result.  The multiply-by-1 follows the LMUL=2 convention where
vector operations use groups of two consecutive registers.

Tile layout (row-major, same as SiLU):
  DRAM tile:  32×32 bf16 = 2048 B stored row-major (first half = rows 0..15,
              second half = rows 16..31 of the tile).
  VMEM_X    = 0x2000  2 KB — current input tile
  VMEM_OUT  = 0x2800  2 KB — current output tile

Constraints:
    - M and N must be multiples of 32.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

VMEM_X = 0x2000
VMEM_OUT = 0x2800

TILE = 32
BF16_BYTES = 2
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES  # 2048


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
            tile = mat[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE].contiguous()
            parts.append(tile.reshape(-1))
    return torch.cat(parts)


def make_type_conversion_instructions(
    M: int,
    N: int,
    dram_x: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate instructions for an M×N bf16 type-conversion (identity copy).

    MRF layout per tile:
        (v0, v1) = input pair
        (v2, v3) = 1.0 constant pair
        (v4, v5) = output pair = input * 1.0 (identity via LMUL=2 vmul)
    """
    assert M % TILE == 0 and N % TILE == 0
    M_tiles = M // TILE
    N_tiles = N // TILE

    insns: list[Instruction] = []
    _emit_load_vmem_addr(1, VMEM_X, insns)    # x1 = VMEM_X
    _emit_load_vmem_addr(2, VMEM_OUT, insns)  # x2 = VMEM_OUT
    _emit_load_imm32(3, TILE_BYTES_BF16, insns)  # x3 = 2048

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    for m in range(M_tiles):
        for n in range(N_tiles):
            x_addr = dram_x + _bf16_tile_offset(m, n, N)
            out_addr = dram_out + _bf16_tile_offset(m, n, N)

            _emit_load_imm32(4, x_addr, insns)
            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=4, rs2=3, channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

            # Load input LMUL=2 pair (v0, v1)
            insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            # Load constant 1.0 LMUL=2 pair (v2, v3)
            insns.append(Instruction("vli.all", VectorArgs(vd=2, imm=1)))  # 1.0
            insns.append(Instruction("delay", ScalarArgs(imm=65)))
            insns.append(Instruction("vli.all", VectorArgs(vd=3, imm=1)))  # 1.0
            insns.append(Instruction("delay", ScalarArgs(imm=65)))

            # LMUL=2: (v4, v5) = (v0, v1) * (v2, v3) = identity
            insns.append(Instruction("vmul.bf16", VectorArgs(vd=4, vs1=0, vs2=2)))
            insns.append(Instruction("delay", ScalarArgs(imm=66)))

            # Store output LMUL=2 pair (v4, v5)
            insns.append(Instruction("vstore", VectorArgs(vd=4, rs1=2, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vstore", VectorArgs(vd=5, rs1=2, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            _emit_load_imm32(5, out_addr, insns)
            insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=5, rs1=2, rs2=3, channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def type_conversion_reference(x: torch.Tensor) -> torch.Tensor:
    """bf16 identity: preserves all bf16 values (bf16 ↔ f32 is lossless in-range)."""
    return x.to(torch.bfloat16)


def _make_program(M: int, N: int, seed: int):
    dram_x = 0x0000
    dram_out = dram_x + M * N * BF16_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, N, dtype=torch.bfloat16)
    expected = type_conversion_reference(x)

    insns = make_type_conversion_instructions(M=M, N=N, dram_x=dram_x, dram_out=dram_out)
    regions = [(dram_x, _tile_matrix_bf16(x, M, N))]
    golden = (dram_out, _tile_matrix_bf16(expected, M, N))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, 32, seed=200)


class ParameterizedTypeConversion32x32Program(Program):
    """bf16 type-conversion on a single 32×32 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64x32_insns, _64x32_regions, _64x32_golden = _make_program(64, 32, seed=201)


class ParameterizedTypeConversion64x32Program(Program):
    """bf16 type-conversion on a 64×32 tensor (2×1 tiles)."""

    instructions: List[Instruction[Any]] = _64x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden


_64x64_insns, _64x64_regions, _64x64_golden = _make_program(64, 64, seed=202)


class ParameterizedTypeConversion64x64Program(Program):
    """bf16 type-conversion on a 64×64 tensor (2×2 tiles)."""

    instructions: List[Instruction[Any]] = _64x64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x64_regions
    golden_result: tuple[int, torch.Tensor] = _64x64_golden
