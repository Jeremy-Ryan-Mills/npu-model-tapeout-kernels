"""Parameterized RMS-norm kernel: y = x * rsqrt(mean(x²) + eps) for M×32.

N is fixed at 32 (the reduction dimension).  M must be a multiple of 32.
Each group of 32 rows is one independent tile processed in col-blocked layout:
  H0 = cols  0–15  (1024 B, bf16 32×16)
  H1 = cols 16–31  (1024 B, bf16 32×16)

DRAM layout (bytes, per _make_program):
  [dram_x     ]  M_groups × 2 × 1024 B  — col-blocked input
  [dram_inv_dim]             1024 B  — 1/32 broadcast constant
  [dram_eps   ]             1024 B  — 1e-6 broadcast constant
  [dram_out   ]  M_groups × 2 × 1024 B  — col-blocked output

VMEM slots (fixed throughout):
  0x2000  VMEM_X_H0   1 KB — current group input H0
  0x2400  VMEM_X_H1   1 KB — current group input H1
  0x3000  VMEM_INVM   2 KB — inv_dim pair (H0 at 0x3000, H1 at 0x3400)
  0x4000  VMEM_EPS    2 KB — eps pair (H0 at 0x4000, H1 at 0x4400)
  0x5000  VMEM_OUT    2 KB — current group output pair

MRF layout per group:
  (v0,  v1 ) = X
  (v2,  v3 ) = X²          via vsquare.bf16
  (v4,  v5 ) = row-sum(X²) via vredsum.row.bf16, then reused for inv_rms
  (v6,  v7 ) = inv_dim (const), then reused for Y = X * inv_rms
  (v8,  v9 ) = eps (const)
  (v10, v11) = mean(X²)     = sum * inv_dim
  (v12, v13) = mean + eps
  (v14, v15) = sqrt(mean + eps)
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

TILE = 32
BF16_BYTES = 2
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 32×16×2 = 1024

VMEM_X = 0x2000       # X H0 at +0, H1 at +1024 (imm12=32)
VMEM_INV_DIM = 0x3000  # inv_dim H0 at 0x3000, H1 at 0x3400
VMEM_EPS = 0x4000     # eps H0 at 0x4000, H1 at 0x4400
VMEM_OUT = 0x5000     # output H0 at 0x5000, H1 at 0x5400


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
    """Arrange M×32 into col-blocked DRAM format: for each 32-row group,
    H0 (cols 0–15) then H1 (cols 16–31), each as a contiguous (32, 16) tile."""
    parts = []
    for g in range(M // TILE):
        group = mat[g * TILE : (g + 1) * TILE, :]
        parts.append(group[:, : TILE // 2].contiguous())
        parts.append(group[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def rms_norm_reference(
    x: torch.Tensor,
    inv_dim: float = 1.0 / 32.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """y = x * rsqrt(mean(x²) + eps).  Mirrors bf16 pair-op ISA sequence."""
    xb = x.to(torch.bfloat16)
    sq = (xb * xb).to(torch.bfloat16)
    row_sum = sq.sum(dim=-1, keepdim=True).to(torch.bfloat16)
    inv_dim_t = torch.full_like(row_sum, inv_dim, dtype=torch.bfloat16)
    eps_t = torch.full_like(row_sum, eps, dtype=torch.bfloat16)
    mean = (row_sum * inv_dim_t).to(torch.bfloat16)
    denom = (mean + eps_t).to(torch.bfloat16)
    root = torch.sqrt(denom.float()).to(torch.bfloat16)
    inv = (1.0 / root.float()).to(torch.bfloat16)
    return (xb * inv).to(x.dtype)


def make_rms_norm_instructions(
    M: int,
    dram_x: int,
    dram_inv_dim: int,
    dram_eps: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate instructions for an M×32 RMS-norm (M must be multiple of 32).

    Scalar register map (fixed for the whole program):
        x1  VMEM_X base     x2  VMEM_INV_DIM base   x3  VMEM_EPS base
        x4  VMEM_OUT base   x5  HALF_BYTES (1024)
        x6  VMEM_X + 1024   x7  VMEM_INV_DIM + 1024
        x8  VMEM_EPS + 1024 x9  VMEM_OUT + 1024
    Per-group scratch: x10, x11  (DRAM src/dst addresses)
    """
    assert M % TILE == 0
    M_groups = M // TILE

    insns: list[Instruction] = []

    # ── one-time scalar setup ─────────────────────────────────────────────
    _emit_load_vmem_addr(1, VMEM_X, insns)
    _emit_load_vmem_addr(2, VMEM_INV_DIM, insns)
    _emit_load_vmem_addr(3, VMEM_EPS, insns)
    _emit_load_vmem_addr(4, VMEM_OUT, insns)
    _emit_load_imm32(5, HALF_BYTES, insns)
    _emit_load_imm32(6, VMEM_X + HALF_BYTES, insns)        # VMEM_X_H1
    _emit_load_imm32(7, VMEM_INV_DIM + HALF_BYTES, insns)  # VMEM_INV_DIM_H1
    _emit_load_imm32(8, VMEM_EPS + HALF_BYTES, insns)      # VMEM_EPS_H1
    _emit_load_imm32(9, VMEM_OUT + HALF_BYTES, insns)      # VMEM_OUT_H1

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # ── load constants once (same inv_dim data into both VMEM halves) ─────
    _emit_load_imm32(10, dram_inv_dim, insns)
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=10, rs2=5, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=7, rs1=10, rs2=5, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    _emit_load_imm32(10, dram_eps, insns)
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=10, rs2=5, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=8, rs1=10, rs2=5, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # ── per-group loop ────────────────────────────────────────────────────
    for g in range(M_groups):
        x_h0 = dram_x + g * 2 * HALF_BYTES
        x_h1 = x_h0 + HALF_BYTES

        _emit_load_imm32(10, x_h0, insns)
        _emit_load_imm32(11, x_h1, insns)

        # DMA X_H0 → VMEM[x1=0x2000], X_H1 → VMEM[x6=0x2400] (parallel)
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=10, rs2=5, channel=0)))
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=6, rs1=11, rs2=5, channel=1)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

        # vload (v0, v1) = X
        insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # vload (v6, v7) = inv_dim (reload each group — v6/v7 reused for output)
        insns.append(Instruction("vload", VectorArgs(vd=6, rs1=2, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=7, rs1=2, imm12=32)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # vload (v8, v9) = eps (reload each group — v8/v9 reused below)
        insns.append(Instruction("vload", VectorArgs(vd=8, rs1=3, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=9, rs1=3, imm12=32)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # (v2, v3) = X²
        insns.append(Instruction("vsquare.bf16", VectorArgs(vd=2, vs1=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v4, v5) = row-sum(X²)
        insns.append(Instruction("vredsum.row.bf16", VectorArgs(vd=4, vs1=2)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v10, v11) = mean(X²) = sum * inv_dim
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=10, vs1=4, vs2=6)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v12, v13) = mean + eps
        insns.append(Instruction("vadd.bf16", VectorArgs(vd=12, vs1=10, vs2=8)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v14, v15) = sqrt(mean + eps)
        insns.append(Instruction("vsqrt.bf16", VectorArgs(vd=14, vs1=12)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # (v4, v5) = inv_rms = 1/sqrt  (reuse pair 4/5)
        insns.append(Instruction("vrecip.bf16", VectorArgs(vd=4, vs1=14)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # (v6, v7) = Y = X * inv_rms  (reuse pair 6/7)
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=6, vs1=0, vs2=4)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # vstore (v6, v7) → VMEM_OUT
        insns.append(Instruction("vstore", VectorArgs(vd=6, rs1=4, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vstore", VectorArgs(vd=7, rs1=4, imm12=32)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # DMA store Y_H0 and Y_H1 (parallel)
        out_h0 = dram_out + g * 2 * HALF_BYTES
        out_h1 = out_h0 + HALF_BYTES
        _emit_load_imm32(10, out_h0, insns)
        _emit_load_imm32(11, out_h1, insns)
        insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=10, rs1=4, rs2=5, channel=0)))
        insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=11, rs1=9, rs2=5, channel=1)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def _make_program(M: int, seed: int):
    M_groups = M // TILE
    dram_x = 0x0000
    dram_inv_dim = dram_x + M_groups * 2 * HALF_BYTES
    dram_eps = dram_inv_dim + HALF_BYTES
    dram_out = dram_eps + HALF_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, TILE, dtype=torch.bfloat16)
    expected = rms_norm_reference(x)

    inv_dim_tile = torch.full((TILE, TILE // 2), 1.0 / TILE, dtype=torch.bfloat16)
    eps_tile = torch.full((TILE, TILE // 2), 1e-6, dtype=torch.bfloat16)

    insns = make_rms_norm_instructions(
        M=M,
        dram_x=dram_x,
        dram_inv_dim=dram_inv_dim,
        dram_eps=dram_eps,
        dram_out=dram_out,
    )
    regions = [
        (dram_x, _colblock_bf16(x, M)),
        (dram_inv_dim, inv_dim_tile),
        (dram_eps, eps_tile),
    ]
    golden = (dram_out, _colblock_bf16(expected, M))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, seed=90)


class ParameterizedRmsNorm32x32Program(Program):
    """RMS-norm on a single 32×32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, seed=91)


class ParameterizedRmsNorm64x32Program(Program):
    """RMS-norm on a 64×32 bf16 tensor (2 groups)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_96_insns, _96_regions, _96_golden = _make_program(96, seed=92)


class ParameterizedRmsNorm96x32Program(Program):
    """RMS-norm on a 96×32 bf16 tensor (3 groups)."""

    instructions: List[Instruction[Any]] = _96_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _96_regions
    golden_result: tuple[int, torch.Tensor] = _96_golden
