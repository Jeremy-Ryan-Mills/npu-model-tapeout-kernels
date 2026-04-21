"""Parameterized GELU (tanh approximation) kernel for M×32 bf16 tensors.

Computes:  y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

N is fixed at 32.  M must be a multiple of 32.  Uses col-blocked layout.
Three polynomial constants (k_coeff, k_sqrt2pi, k_half) are loaded from
DRAM once; k_one = 1.0 is materialised with vli.all.

DRAM layout (per _make_program):
  [dram_x        ]  M_groups × 2 × 1024 B  — col-blocked input
  [dram_k_coeff  ]  1024 B  — 0.044715 broadcast tile (32×16 bf16)
  [dram_k_sqrt2pi]  1024 B  — sqrt(2/π) ≈ 0.7979 broadcast tile
  [dram_k_half   ]  1024 B  — 0.5 broadcast tile
  [dram_out      ]  M_groups × 2 × 1024 B  — col-blocked output

VMEM slots:
  0x2000  VMEM_X        2 KB — current input pair
  0x2800  VMEM_K_COEFF  2 KB — 0.044715 pair (both halves same data)
  0x3000  VMEM_K_SQRT   2 KB — sqrt(2/π) pair
  0x3800  VMEM_K_HALF   2 KB — 0.5 pair
  0x4000  VMEM_OUT      2 KB — current output pair

MRF layout per group:
  (v0,  v1 ) = X
  (v2,  v3 ) = X²              via vsquare.bf16
  (v12, v13) = X³              via vmul(v2, v0)
  (v12, v13) = k_coeff * X³   via vmul(v12, v4)
  (v12, v13) = X + k_coeff*X³ via vadd(v0, v12)
  (v12, v13) *= k_sqrt2pi      via vmul(v12, v6)
  (v12, v13) = tanh(...)       via vtanh.bf16
  (v10, v11) = 1.0             via vli.all
  (v12, v13) = 1 + tanh(...)   via vadd(v10, v12)
  (v12, v13) *= k_half         via vmul(v12, v8)
  (v14, v15) = Y = x*(...)     via vmul(v0, v12)
"""

import math
from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, ScalarArgs, VectorArgs

TILE = 32
BF16_BYTES = 2
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES  # 1024

VMEM_X = 0x2000
VMEM_K_COEFF = 0x2800
VMEM_K_SQRT = 0x3000
VMEM_K_HALF = 0x3800
VMEM_OUT = 0x4000

_K_COEFF = 0.044715
_K_SQRT2PI = math.sqrt(2.0 / math.pi)  # ≈ 0.7979
_K_HALF = 0.5


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


def gelu_tanh_reference(x: torch.Tensor) -> torch.Tensor:
    """GELU tanh approximation matching the bf16 ISA sequence."""
    xf = x.float()
    x3 = xf.pow(3.0)
    inner = _K_SQRT2PI * (xf + _K_COEFF * x3)
    return (xf * 0.5 * (1.0 + torch.tanh(inner))).to(x.dtype)


def make_gelu_tanh_instructions(
    M: int,
    dram_x: int,
    dram_k_coeff: int,
    dram_k_sqrt2pi: int,
    dram_k_half: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate instructions for an M×32 GELU (tanh approximation).

    Scalar register map:
        x1  VMEM_X base        x2  VMEM_K_COEFF base   x3  VMEM_K_SQRT base
        x4  VMEM_K_HALF base   x5  VMEM_OUT base        x6  HALF_BYTES (1024)
        x7  VMEM_X + 1024      (X_H1 VMEM dest)
        x8  VMEM_K_COEFF+1024  x9  VMEM_K_SQRT+1024   x10  VMEM_K_HALF+1024
        x11  VMEM_OUT + 1024
    Per-group scratch: x12, x13
    """
    assert M % TILE == 0
    M_groups = M // TILE

    insns: list[Instruction] = []

    _emit_load_vmem_addr(1, VMEM_X, insns)
    _emit_load_vmem_addr(2, VMEM_K_COEFF, insns)
    _emit_load_vmem_addr(3, VMEM_K_SQRT, insns)
    _emit_load_vmem_addr(4, VMEM_K_HALF, insns)
    _emit_load_vmem_addr(5, VMEM_OUT, insns)
    _emit_load_imm32(6, HALF_BYTES, insns)
    _emit_load_imm32(7, VMEM_X + HALF_BYTES, insns)
    _emit_load_imm32(8, VMEM_K_COEFF + HALF_BYTES, insns)
    _emit_load_imm32(9, VMEM_K_SQRT + HALF_BYTES, insns)
    _emit_load_imm32(10, VMEM_K_HALF + HALF_BYTES, insns)
    _emit_load_imm32(11, VMEM_OUT + HALF_BYTES, insns)

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    # Load constants once (same data into both halves of each pair)
    _emit_load_imm32(12, dram_k_coeff, insns)
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=12, rs2=6, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=8, rs1=12, rs2=6, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    _emit_load_imm32(12, dram_k_sqrt2pi, insns)
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=12, rs2=6, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=9, rs1=12, rs2=6, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    _emit_load_imm32(12, dram_k_half, insns)
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=4, rs1=12, rs2=6, channel=0)))
    insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=10, rs1=12, rs2=6, channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    for g in range(M_groups):
        x_h0 = dram_x + g * 2 * HALF_BYTES
        x_h1 = x_h0 + HALF_BYTES

        _emit_load_imm32(12, x_h0, insns)
        _emit_load_imm32(13, x_h1, insns)

        # DMA X_H0 → VMEM_X, X_H1 → VMEM_X+1024 (parallel)
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=12, rs2=6, channel=0)))
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=7, rs1=13, rs2=6, channel=1)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

        # vload (v0, v1) = X
        insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # vload constants (reload each group to restore after v4/v6/v8 clobber)
        insns.append(Instruction("vload", VectorArgs(vd=4, rs1=2, imm12=0)))   # k_coeff H0
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=5, rs1=2, imm12=32)))  # k_coeff H1
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=6, rs1=3, imm12=0)))   # k_sqrt H0
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=7, rs1=3, imm12=32)))  # k_sqrt H1
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=8, rs1=4, imm12=0)))   # k_half H0
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vload", VectorArgs(vd=9, rs1=4, imm12=32)))  # k_half H1
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # (v2, v3) = X²
        insns.append(Instruction("vsquare.bf16", VectorArgs(vd=2, vs1=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v12, v13) = X³ = X² * X
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=12, vs1=2, vs2=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v12, v13) = k_coeff * X³
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=12, vs1=12, vs2=4)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v12, v13) = X + k_coeff*X³
        insns.append(Instruction("vadd.bf16", VectorArgs(vd=12, vs1=0, vs2=12)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v12, v13) = sqrt(2/π) * (X + k_coeff*X³)
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=12, vs1=12, vs2=6)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v12, v13) = tanh(...)
        insns.append(Instruction("vtanh.bf16", VectorArgs(vd=12, vs1=12)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # (v10, v11) = 1.0  — vli.all is a single-register write; two calls needed
        insns.append(Instruction("vli.all", VectorArgs(vd=10, imm=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=2)))
        insns.append(Instruction("vli.all", VectorArgs(vd=11, imm=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=2)))

        # (v12, v13) = 1 + tanh(...)
        insns.append(Instruction("vadd.bf16", VectorArgs(vd=12, vs1=10, vs2=12)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v12, v13) = 0.5 * (1 + tanh(...))
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=12, vs1=12, vs2=8)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # (v14, v15) = Y = X * 0.5*(1+tanh(...))
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=14, vs1=0, vs2=12)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # vstore (v14, v15) → VMEM_OUT
        insns.append(Instruction("vstore", VectorArgs(vd=14, rs1=5, imm12=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vstore", VectorArgs(vd=15, rs1=5, imm12=32)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # DMA store Y_H0, Y_H1 (parallel)
        out_h0 = dram_out + g * 2 * HALF_BYTES
        out_h1 = out_h0 + HALF_BYTES
        _emit_load_imm32(12, out_h0, insns)
        _emit_load_imm32(13, out_h1, insns)
        insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=12, rs1=5, rs2=6, channel=0)))
        insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=13, rs1=11, rs2=6, channel=1)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def _make_program(M: int, seed: int):
    M_groups = M // TILE
    dram_x = 0x0000
    dram_k_coeff = dram_x + M_groups * 2 * HALF_BYTES
    dram_k_sqrt2pi = dram_k_coeff + HALF_BYTES
    dram_k_half = dram_k_sqrt2pi + HALF_BYTES
    dram_out = dram_k_half + HALF_BYTES

    torch.manual_seed(seed)
    x = torch.randn(M, TILE, dtype=torch.bfloat16) * 0.5
    expected = gelu_tanh_reference(x)

    k_coeff_tile = torch.full((TILE, TILE // 2), _K_COEFF, dtype=torch.bfloat16)
    k_sqrt_tile = torch.full((TILE, TILE // 2), _K_SQRT2PI, dtype=torch.bfloat16)
    k_half_tile = torch.full((TILE, TILE // 2), _K_HALF, dtype=torch.bfloat16)

    insns = make_gelu_tanh_instructions(
        M=M,
        dram_x=dram_x,
        dram_k_coeff=dram_k_coeff,
        dram_k_sqrt2pi=dram_k_sqrt2pi,
        dram_k_half=dram_k_half,
        dram_out=dram_out,
    )
    regions = [
        (dram_x, _colblock_bf16(x, M)),
        (dram_k_coeff, k_coeff_tile),
        (dram_k_sqrt2pi, k_sqrt_tile),
        (dram_k_half, k_half_tile),
    ]
    golden = (dram_out, _colblock_bf16(expected, M))
    return insns, regions, golden


_32_insns, _32_regions, _32_golden = _make_program(32, seed=140)


class ParameterizedGeluTanh32x32Program(Program):
    """GELU (tanh approximation) on a single 32×32 bf16 tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


_64_insns, _64_regions, _64_golden = _make_program(64, seed=141)


class ParameterizedGeluTanh64x32Program(Program):
    """GELU (tanh approximation) on a 64×32 bf16 tensor (2 groups)."""

    instructions: List[Instruction[Any]] = _64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _64_regions
    golden_result: tuple[int, torch.Tensor] = _64_golden


_96_insns, _96_regions, _96_golden = _make_program(96, seed=142)


class ParameterizedGeluTanh96x32Program(Program):
    """GELU (tanh approximation) on a 96×32 bf16 tensor (3 groups)."""

    instructions: List[Instruction[Any]] = _96_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _96_regions
    golden_result: tuple[int, torch.Tensor] = _96_golden
