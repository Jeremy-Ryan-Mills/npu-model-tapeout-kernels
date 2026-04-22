"""Parameterized bf16→fp8 requantization kernel for arbitrary M×N.

Reads col-blocked bf16 input tiles (32×16 halves) and packs them into
contiguous 32×32 fp8 output tiles via vpack.bf16.fp8 with unit scale.

Constraints:
    - M and N must be multiples of 32.

DRAM layout (per _make_program):
  [dram_x  ]  M_tiles × N_tiles × 2 × 1024 B  — col-blocked bf16 input
  [dram_out ]  M_tiles × N_tiles     × 1024 B  — fp8 output (32×32 × 1 B)

VMEM slots:
  0x2000  VMEM_X_H0  1 KB — bf16 H0 (cols 0–15)
  0x2400  VMEM_X_H1  1 KB — bf16 H1 (cols 16–31)
  0x3000  VMEM_OUT   1 KB — fp8 tile (32×32 × 1 B)
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

TILE = 32
BF16_BYTES = 2
FP8_BYTES = 1
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 1024  (32×16 bf16)
FP8_TILE_BYTES = TILE * TILE * FP8_BYTES       # 1024  (32×32 fp8)

VMEM_X_H0 = 0x2000
VMEM_X_H1 = 0x2400
VMEM_OUT = 0x3000


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


def _tile_fp8(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Arrange M×N fp8 tensor into row-major tile order (matching DRAM output)."""
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for mt in range(M_tiles):
        for nt in range(N_tiles):
            tile = mat[mt * TILE : (mt + 1) * TILE, nt * TILE : (nt + 1) * TILE].contiguous()
            parts.append(tile.reshape(-1))
    return torch.cat(parts)


def _colblock_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Arrange M×N into tiled col-blocked format.

    For tile (m, n): H0 (cols n*32 to n*32+15) then H1 (cols n*32+16 to n*32+31),
    each as a contiguous (32, 16) bf16 slice.  Tile order: row-major over tiles.
    """
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for mt in range(M_tiles):
        for nt in range(N_tiles):
            rows = mat[mt * TILE : (mt + 1) * TILE, nt * TILE : (nt + 1) * TILE]
            parts.append(rows[:, : TILE // 2].contiguous())
            parts.append(rows[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def requant_reference(x: torch.Tensor) -> torch.Tensor:
    """bf16 → fp8_e4m3fn unit-scale cast.  Matches seli imm=1 path."""
    return x.to(torch.float8_e4m3fn)


def make_requant_instructions(
    M: int,
    N: int,
    dram_x: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate instructions for an M×N bf16→fp8 requantization.

    Scalar register map:
        x1  VMEM_X_H0     x2  VMEM_X_H1     x3  VMEM_OUT
        x4  HALF_BYTES (1024)
    ERF: seli rd=0, imm=1  → scale register 0 = 1.0 (unit scale)
    Per-tile scratch: x5, x6, x7
    """
    assert M % TILE == 0 and N % TILE == 0
    M_tiles = M // TILE
    N_tiles = N // TILE

    insns: list[Instruction] = []

    _emit_load_vmem_addr(1, VMEM_X_H0, insns)
    _emit_load_vmem_addr(2, VMEM_X_H1, insns)
    _emit_load_vmem_addr(3, VMEM_OUT, insns)
    _emit_load_imm32(4, HALF_BYTES, insns)

    # Set unit scale in ERF register 0 once
    insns.append(Instruction("seli", ScalarArgs(rd=0, imm=1)))

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    tile_idx = 0
    for mt in range(M_tiles):
        for nt in range(N_tiles):
            h0_addr = dram_x + tile_idx * 2 * HALF_BYTES
            h1_addr = h0_addr + HALF_BYTES
            out_addr = dram_out + tile_idx * FP8_TILE_BYTES

            _emit_load_imm32(5, h0_addr, insns)
            _emit_load_imm32(6, h1_addr, insns)

            # DMA H0 → VMEM_X_H0, H1 → VMEM_X_H1 (parallel)
            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=5, rs2=4, channel=0)))
            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=6, rs2=4, channel=1)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

            # vload v0 = H0, v1 = H1  (single-register loads)
            insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=1, rs1=2, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            # Pack (v0, v1) bf16 pair → v2 fp8 tile using scale ERF[0]=1
            insns.append(Instruction("vpack.bf16.fp8", VectorArgs(vd=2, vs1=0, es1=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=66)))

            # vstore v2 → VMEM_OUT (fp8, 1024 B)
            insns.append(Instruction("vstore", VectorArgs(vd=2, rs1=3, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            # DMA store fp8 tile → DRAM
            _emit_load_imm32(7, out_addr, insns)
            insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=7, rs1=3, rs2=4, channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

            tile_idx += 1

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def _make_program(M: int, N: int, seed: int):
    M_tiles = M // TILE
    N_tiles = N // TILE
    n_tiles = M_tiles * N_tiles

    dram_x = 0x0000
    dram_out = dram_x + n_tiles * 2 * HALF_BYTES

    torch.manual_seed(seed)
    # Keep values in fp8_e4m3fn range
    x = torch.randn(M, N, dtype=torch.bfloat16) * 0.5
    expected = requant_reference(x)

    insns = make_requant_instructions(M=M, N=N, dram_x=dram_x, dram_out=dram_out)
    regions = [(dram_x, _colblock_bf16(x, M, N))]
    golden = (dram_out, _tile_fp8(expected, M, N))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, 32, seed=120)


class ParameterizedRequant32x32Program(Program):
    """bf16→fp8 requant on a single 32×32 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, 64, seed=121)


class ParameterizedRequant64x64Program(Program):
    """bf16→fp8 requant on a 64×64 tensor (2×2 tiles)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_64x32_insns, _64x32_regions, _64x32_golden = _make_program(64, 32, seed=122)


class ParameterizedRequant64x32Program(Program):
    """bf16→fp8 requant on a 64×32 tensor (2×1 tiles)."""

    instructions: List[Instruction[Any]] = _64x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
