"""Parameterized int8-matmul kernel: C = A_i8 @ B_i8 for M×K×N.

Gemma Expert uses int8 as a fallback because its hidden dim (720) is not
a multiple of 32, preventing the fp8/MX block format.  On the NPU the
int8 weights are stored in the same 8-bit MRF layout as fp8 weights;
small integer values (–8…7) are exactly representable in fp8_e4m3fn, so
the same MXU instructions handle both paths.

ISA is identical to parameterized_matmul.py.  The distinction is in the
data provenance (integer weights vs floating-point weights) and the
reference function (integer accumulation semantics).

DRAM layout:
  [dram_a]  M_tiles × K_tiles × 1024 B  — fp8-packed int8 A tiles
  [dram_b]  K_tiles × N_tiles × 1024 B  — fp8-packed int8 B tiles
  [dram_c]  M_tiles × N_tiles × 2048 B  — col-blocked bf16 output

VMEM slots (reused per tile):
  0x2000  VMEM_A   1 KB
  0x2400  VMEM_B   1 KB
  0x2800  VMEM_C0  1 KB — C low  half
  0x2C00  VMEM_C1  1 KB — C high half

Constraints: M, K, N multiples of 32.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs

# Reuse the instruction generator and helpers from parameterized_matmul
from .parameterized_matmul import (
    make_matmul_instructions,
    _tile_matrix,
    _expected_stacked,
)

TILE = 32
FP8_BYTES = 1
BF16_BYTES = 2


def matmul_i8_reference(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Int8 matmul reference: A @ B with fp8-level precision.

    Inputs are stored as fp8 (int8 values in –7…7 are exactly representable).
    Accumulation is float16 per K-tile (matches MXU bf16 accumulator semantics).
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    K_tiles = K // TILE

    acc = None
    for k in range(K_tiles):
        a_k = a[:, k * TILE : (k + 1) * TILE].to(torch.float16)
        b_k = b[k * TILE : (k + 1) * TILE, :].to(torch.float16)
        if acc is None:
            acc = a_k @ b_k
        else:
            acc = acc.to(torch.bfloat16).to(torch.float16) + (a_k @ b_k)
    return acc.to(torch.bfloat16)


def _make_program(M: int, K: int, N: int, seed: int):
    dram_a = 0x0000
    dram_b = dram_a + M * K * FP8_BYTES
    dram_c = dram_b + K * N * FP8_BYTES

    torch.manual_seed(seed)
    # Int8 values in –7…7 are exactly representable in fp8_e4m3fn.
    a_i8 = torch.randint(-7, 7, (M, K), dtype=torch.int8)
    b_i8 = torch.randint(-7, 7, (K, N), dtype=torch.int8)
    # Pack into fp8 format for DRAM storage
    a = a_i8.float().to(torch.float8_e4m3fn)
    b = b_i8.float().to(torch.float8_e4m3fn)
    expected = matmul_i8_reference(a, b)

    insns = make_matmul_instructions(M=M, K=K, N=N, dram_a=dram_a, dram_b=dram_b, dram_c=dram_c)
    regions = [(dram_a, _tile_matrix(a, M, K)), (dram_b, _tile_matrix(b, K, N))]
    golden = (dram_c, _expected_stacked(expected, M, N))
    return insns, regions, golden


# ── 32×32×32: single tile ─────────────────────────────────────────────────────

_32_insns, _32_regions, _32_golden = _make_program(32, 32, 32, seed=400)


class ParameterizedMatmulI832x32x32Program(Program):
    """Int8 matmul 32×32×32 — 1×1 output tile, 1 K-tile."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden


# ── 32×64×32: single output tile, 2 K-tiles ──────────────────────────────────

_kchain_insns, _kchain_regions, _kchain_golden = _make_program(32, 64, 32, seed=401)


class ParameterizedMatmulI832x64x32Program(Program):
    """Int8 matmul 32×64×32 — K-accumulation path (2 K-tiles)."""

    instructions: List[Instruction[Any]] = _kchain_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _kchain_regions
    golden_result: tuple[int, torch.Tensor] = _kchain_golden


# ── 64×32×64: 4 output tiles ──────────────────────────────────────────────────

_multi_insns, _multi_regions, _multi_golden = _make_program(64, 32, 64, seed=402)


class ParameterizedMatmulI864x32x64Program(Program):
    """Int8 matmul 64×32×64 — 2×2 output tiles."""

    instructions: List[Instruction[Any]] = _multi_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _multi_regions
    golden_result: tuple[int, torch.Tensor] = _multi_golden
