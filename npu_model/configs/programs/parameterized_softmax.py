"""Parameterized row-wise stable softmax kernel for M×32 bf16 tensors.

N is fixed at 32.  M must be a multiple of 32.  Each group of 32 rows is
processed independently using the col-blocked layout:
  H0 = cols  0–15  (1024 B bf16 32×16)
  H1 = cols 16–31  (1024 B bf16 32×16)

DRAM layout (per _make_program):
  [dram_x  ]  M_groups × 2 × 1024 B  — col-blocked input
  [dram_out ]  M_groups × 2 × 1024 B  — col-blocked output

VMEM slots:
  0x2000  VMEM_X   2 KB — current input pair  (H0 at +0, H1 at +1024)
  0x3000  VMEM_OUT 2 KB — current output pair (H0 at +0, H1 at +1024)

MRF layout per group:
  (v0,  v1 ) = X
  (v2,  v3 ) = rowmax(X)   via vredmax.row.bf16
  (v4,  v5 ) = X − rowmax  via vsub.bf16
  (v6,  v7 ) = exp(X−max)  via vexp.bf16
  (v8,  v9 ) = rowsum(exp) via vredsum.row.bf16
  (v10, v11) = 1/rowsum    via vrecip.bf16
  (v12, v13) = Y = exp * inv_rowsum
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


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """Row-wise stable softmax matching the bf16 ISA sequence."""
    xf = x.float()
    xm = xf - xf.max(dim=-1, keepdim=True).values
    ex = xm.exp()
    return (ex / ex.sum(dim=-1, keepdim=True)).to(x.dtype)


def make_softmax_instructions(
    M: int,
    dram_x: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate instructions for an M×32 row-wise softmax.

    Scalar register map:
        x1  VMEM_X base   x2  VMEM_OUT base   x3  HALF_BYTES (1024)
        x4  VMEM_X + 1024 (X_H1 VMEM dest)   x5  VMEM_OUT + 1024 (for dma.store H1)
    Per-group scratch: x6, x7
    """
    assert M % TILE == 0
    M_groups = M // TILE

    insns: list[Instruction] = []

    _emit_load_vmem_addr(1, VMEM_X, insns)
    _emit_load_vmem_addr(2, VMEM_OUT, insns)
    _emit_load_imm32(3, HALF_BYTES, insns)
    _emit_load_imm32(4, VMEM_X + HALF_BYTES, insns)    # VMEM_X_H1
    _emit_load_imm32(5, VMEM_OUT + HALF_BYTES, insns)  # VMEM_OUT_H1

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    for g in range(M_groups):
        x_h0 = dram_x + g * 2 * HALF_BYTES
        x_h1 = x_h0 + HALF_BYTES

        _emit_load_imm32(6, x_h0, insns)
        _emit_load_imm32(7, x_h1, insns)

        # DMA X_H0 → VMEM[x1], X_H1 → VMEM[x4] (parallel)
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=6, rs2=3, channel=0)))
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=7, rs2=3, channel=1)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

        # vload (v0, v1) = X
        insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # (v2, v3) = rowmax(X)
        insns.append(Instruction("vredmax.row.bf16", VectorArgs(vd=2, vs1=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=69)))

        # (v4, v5) = X − rowmax
        insns.append(Instruction("vsub.bf16", VectorArgs(vd=4, vs1=0, vs2=2)))
        insns.append(Instruction("delay", ScalarArgs(imm=66)))

        # (v6, v7) = exp(X − rowmax)
        insns.append(Instruction("vexp.bf16", VectorArgs(vd=6, vs1=4)))
        insns.append(Instruction("delay", ScalarArgs(imm=66)))

        # (v8, v9) = rowsum(exp)
        insns.append(Instruction("vredsum.row.bf16", VectorArgs(vd=8, vs1=6)))
        insns.append(Instruction("delay", ScalarArgs(imm=69)))

        # (v10, v11) = 1/rowsum
        insns.append(Instruction("vrecip.bf16", VectorArgs(vd=10, vs1=8)))
        insns.append(Instruction("delay", ScalarArgs(imm=66)))

        # (v12, v13) = exp * inv_rowsum = Y
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=12, vs1=6, vs2=10)))
        insns.append(Instruction("delay", ScalarArgs(imm=66)))

        # vstore (v12, v13) → VMEM_OUT
        insns.append(Instruction("vstore", VectorArgs(vd=12, rs1=2, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vstore", VectorArgs(vd=13, rs1=2, imm12=32)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # DMA store Y_H0 and Y_H1 (parallel)
        out_h0 = dram_out + g * 2 * HALF_BYTES
        out_h1 = out_h0 + HALF_BYTES
        _emit_load_imm32(6, out_h0, insns)
        _emit_load_imm32(7, out_h1, insns)
        insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=6, rs1=2, rs2=3, channel=0)))
        insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=7, rs1=5, rs2=3, channel=1)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def _make_program(M: int, seed: int):
    M_groups = M // TILE
    dram_x = 0x0000
    dram_out = dram_x + M_groups * 2 * HALF_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, TILE, dtype=torch.bfloat16) * 2.0
    expected = softmax_reference(x)

    insns = make_softmax_instructions(M=M, dram_x=dram_x, dram_out=dram_out)
    regions = [(dram_x, _colblock_bf16(x, M))]
    golden = (dram_out, _colblock_bf16(expected, M))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, seed=100)


class ParameterizedSoftmax32x32Program(Program):
    """Row-wise softmax on a single 32×32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, seed=101)


class ParameterizedSoftmax64x32Program(Program):
    """Row-wise softmax on a 64×32 bf16 tensor (2 groups)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_96_insns, _96_regions, _96_golden = _make_program(96, seed=102)


class ParameterizedSoftmax96x32Program(Program):
    """Row-wise softmax on a 96×32 bf16 tensor (3 groups)."""

    instructions: List[Instruction[Any]] = _96_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _96_regions
    golden_result: tuple[int, torch.Tensor] = _96_golden
