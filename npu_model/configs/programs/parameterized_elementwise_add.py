"""Parameterized elementwise add kernel: C = A + B for arbitrary M×N.

Tiles both inputs and output into 32×32 bf16 blocks.  Each block is
stored row-major in DRAM (2048 B contiguous), matching the layout of a
plain torch.Tensor with dtype=bfloat16.

Constraints:
    - M and N must be multiples of 32.
    - A, B, and C are stored in DRAM in tiled layout:
      tile (m, n) starts at byte (m * N_tiles + n) * 2048.

VMEM uses three fixed slots (reused across all tiles):
    VMEM_A  = 0x2000   2 KB — current A tile
    VMEM_B  = 0x2800   2 KB — current B tile
    VMEM_C  = 0x3000   2 KB — current C tile
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

# ── fixed VMEM addresses ──────────────────────────────────────────────────
VMEM_A = 0x2000
VMEM_B = 0x2800
VMEM_C = 0x3000

TILE = 32
BF16_BYTES = 2
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES  # 2048


# ── helpers ───────────────────────────────────────────────────────────────

def _emit_load_imm32(rd: int, value: int, out: list[Instruction]) -> None:
    """Emit lui + addi to materialise an arbitrary 32-bit signed value in rd."""
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
    """Byte offset of bf16 tile (m, n) in tiled DRAM layout."""
    return (m * (N // TILE) + n) * TILE_BYTES_BF16


def _tile_matrix_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Rearrange (M, N) row-major bf16 tensor into tiled DRAM layout.

    Each 32×32 tile is stored contiguously in row-major order.
    """
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for r in range(M_tiles):
        for c in range(N_tiles):
            tile = mat[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE].contiguous()
            parts.append(tile.reshape(-1))
    return torch.cat(parts)


# ── instruction sequence generator ───────────────────────────────────────

def make_elementwise_add_instructions(
    M: int,
    N: int,
    dram_a: int,
    dram_b: int,
    dram_c: int,
) -> list[Instruction]:
    """Generate the full instruction list for an M×N elementwise add.

    Scalar register allocation:
        x1  VMEM_A address       x2  VMEM_B address
        x3  VMEM_C address       x4  transfer size (2048)
        x5  scratch: DRAM A tile address
        x6  scratch: DRAM B tile address
        x7  scratch: DRAM C tile address
    """
    assert M % TILE == 0 and N % TILE == 0, (
        f"M={M}, N={N} must both be multiples of {TILE}"
    )
    M_tiles = M // TILE
    N_tiles = N // TILE

    insns: list[Instruction] = []

    # One-time scalar setup
    _emit_load_vmem_addr(1, VMEM_A, insns)
    _emit_load_vmem_addr(2, VMEM_B, insns)
    _emit_load_vmem_addr(3, VMEM_C, insns)
    _emit_load_imm32(4, TILE_BYTES_BF16, insns)  # x4 = 2048

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

            # DMA A and B in parallel
            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=5, rs2=4, channel=0)))
            insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=6, rs2=4, channel=1)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

            # Load both halves of A (v0, v1) and B (v2, v3)
            insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=2, rs1=2, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vload", VectorArgs(vd=3, rs1=2, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            # Elementwise add: (v4, v5) = (v0+v2, v1+v3)  — pair-op writes both
            insns.append(Instruction("vadd.bf16", VectorArgs(vd=4, vs1=0, vs2=2)))
            insns.append(Instruction("delay", ScalarArgs(imm=4)))

            # Store to VMEM_C then DMA to DRAM
            insns.append(Instruction("vstore", VectorArgs(vd=4, rs1=3, imm12=0)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vstore", VectorArgs(vd=5, rs1=3, imm12=32)))
            insns.append(Instruction("delay", ScalarArgs(imm=20)))

            _emit_load_imm32(7, c_addr, insns)
            insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=7, rs1=3, rs2=4, channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


# ── reference implementation ──────────────────────────────────────────────

def elementwise_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() + b.float()).to(a.dtype)


# ── program factory ───────────────────────────────────────────────────────

def _make_program(M: int, N: int, seed: int):
    """Build (instructions, memory_regions, golden_result) for an M×N add."""
    dram_a = 0x0000
    dram_b = dram_a + M * N * BF16_BYTES
    dram_c = dram_b + M * N * BF16_BYTES

    torch.manual_seed(seed)
    a = torch.randn(M, N, dtype=torch.bfloat16)
    b = torch.randn(M, N, dtype=torch.bfloat16)
    expected = elementwise_add_reference(a, b)

    insns = make_elementwise_add_instructions(M=M, N=N, dram_a=dram_a, dram_b=dram_b, dram_c=dram_c)
    regions = [
        (dram_a, _tile_matrix_bf16(a, M, N)),
        (dram_b, _tile_matrix_bf16(b, M, N)),
    ]
    golden = (dram_c, _tile_matrix_bf16(expected, M, N))
    return insns, regions, golden


# ── 32×32: single tile ────────────────────────────────────────────────────

_32_insns, _32_regions, _32_golden = _make_program(32, 32, seed=10)


class ParameterizedElementwiseAdd32x32Program(Program):
    """Elementwise add on a single 32×32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


# ── 64×64: 2×2 tiles ─────────────────────────────────────────────────────

_64_insns, _64_regions, _64_golden = _make_program(64, 64, seed=11)


class ParameterizedElementwiseAdd64x64Program(Program):
    """Elementwise add on a 64×64 bf16 tensor (2×2 tiles)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


# ── 64×32: 2×1 tiles ─────────────────────────────────────────────────────

_64x32_insns, _64x32_regions, _64x32_golden = _make_program(64, 32, seed=12)


class ParameterizedElementwiseAdd64x32Program(Program):
    """Elementwise add on a 64×32 bf16 tensor (2×1 tiles)."""

    instructions: List[Instruction[Any]] = _64x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64x32_regions
    golden_result: tuple[int, torch.Tensor] = _64x32_golden
