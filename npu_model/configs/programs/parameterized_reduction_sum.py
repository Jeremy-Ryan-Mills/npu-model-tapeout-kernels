"""Parameterized row-wise reduction-sum kernel for M×32 bf16 tensors.

N is fixed at 32.  M must be a multiple of 32.  For each 32-row group,
vredsum.row.bf16 reduces all 32 columns and broadcasts the scalar row-sum
back across both halves.  Only H0 of the result (the first 32×16 half) is
stored; H1 is identical under the broadcast.

DRAM layout (per _make_program):
  [dram_x  ]  M_groups × 2 × 1024 B  — col-blocked input (H0 then H1 per group)
  [dram_out ]  M_groups     × 1024 B  — output row-sums (32×16 per group)

VMEM slots:
  0x2000  VMEM_X   2 KB — current input pair  (H0 at +0, H1 at +1024)
  0x3000  VMEM_OUT 1 KB — output (v4 only, 32×16)

MRF layout per group:
  (v0, v1) = X         via vload pair
  (v4, v5) = rowsum(X) via vredsum.row.bf16  [H0 == H1 under broadcast]
  → store v4 → VMEM_OUT → DMA to dram_out
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

TILE = 32
BF16_BYTES = 2
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 1024

VMEM_X = 0x2000
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


def _colblock_bf16(mat: torch.Tensor, M: int) -> torch.Tensor:
    """Arrange M×32 into col-blocked format: for each 32-row group,
    H0 (cols 0–15) then H1 (cols 16–31)."""
    parts = []
    for g in range(M // TILE):
        group = mat[g * TILE : (g + 1) * TILE, :]
        parts.append(group[:, : TILE // 2].contiguous())
        parts.append(group[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def reduction_sum_reference(x: torch.Tensor) -> torch.Tensor:
    """Row-wise sum, broadcast to 16 columns.  Mirrors vredsum.row.bf16."""
    row_sum = x.sum(dim=-1, keepdim=True).to(torch.bfloat16)
    return row_sum.expand(-1, TILE // 2).contiguous().to(x.dtype)


def make_reduction_sum_instructions(
    M: int,
    dram_x: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate instructions for an M×32 row-wise reduction-sum.

    Scalar register map:
        x1  VMEM_X base       x2  VMEM_OUT base    x3  HALF_BYTES (1024)
        x4  VMEM_X + 1024     (X_H1 VMEM dest)
    Per-group scratch: x5, x6
    """
    assert M % TILE == 0
    M_groups = M // TILE

    insns: list[Instruction] = []

    _emit_load_vmem_addr(1, VMEM_X, insns)
    _emit_load_vmem_addr(2, VMEM_OUT, insns)
    _emit_load_imm32(3, HALF_BYTES, insns)
    _emit_load_imm32(4, VMEM_X + HALF_BYTES, insns)  # VMEM_X_H1

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    for g in range(M_groups):
        x_h0 = dram_x + g * 2 * HALF_BYTES
        x_h1 = x_h0 + HALF_BYTES

        _emit_load_imm32(5, x_h0, insns)
        _emit_load_imm32(6, x_h1, insns)

        # DMA X_H0 → VMEM[x1], X_H1 → VMEM[x4] (parallel)
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=5, rs2=3, channel=0)))
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=6, rs2=3, channel=1)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

        # vload (v0, v1) = X pair
        insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # (v4, v5) = rowsum(X) broadcast — pair-op, H0 == H1
        insns.append(Instruction("vredsum.row.bf16", VectorArgs(vd=4, vs1=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # Store v4 (H0 of result) → VMEM_OUT
        insns.append(Instruction("vstore", VectorArgs(vd=4, rs1=2, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # DMA store 1024 B from VMEM_OUT → dram_out group slot
        _emit_load_imm32(5, dram_out + g * HALF_BYTES, insns)
        insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=5, rs1=2, rs2=3, channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def _make_program(M: int, seed: int):
    M_groups = M // TILE
    dram_x = 0x0000
    dram_out = dram_x + M_groups * 2 * HALF_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, TILE, dtype=torch.bfloat16)
    expected = reduction_sum_reference(x)  # (M, 16) bf16

    insns = make_reduction_sum_instructions(M=M, dram_x=dram_x, dram_out=dram_out)
    regions = [(dram_x, _colblock_bf16(x, M))]
    golden = (dram_out, expected)
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, seed=110)


class ParameterizedReductionSum32x32Program(Program):
    """Row-wise reduction-sum on a single 32×32 bf16 tile → (32, 16) output."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, seed=111)


class ParameterizedReductionSum64x32Program(Program):
    """Row-wise reduction-sum on a 64×32 bf16 tensor (2 groups) → (64, 16) output."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_96_insns, _96_regions, _96_golden = _make_program(96, seed=112)


class ParameterizedReductionSum96x32Program(Program):
    """Row-wise reduction-sum on a 96×32 bf16 tensor (3 groups) → (96, 16) output."""

    instructions: List[Instruction[Any]] = _96_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _96_regions
    golden_result: tuple[int, torch.Tensor] = _96_golden
