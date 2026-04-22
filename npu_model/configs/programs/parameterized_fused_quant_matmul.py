"""Parameterized fused-quant-matmul kernel: fp8(A_bf16) @ B_fp8 for M×K×N.

Fuses int8_quantize (bf16 → fp8 unit-scale cast) with a tiled fp8 matmul.
In SmolVLA this pattern appears 30× as fused_quant_matmul (int8_quantize +
linalg.matmul), where bf16 activations from a previous layer are quantized
on-the-fly before being fed to the MXU.

Per K-tile of A:
  1. Load bf16 A tile (col-blocked, 2 × 1024 B) → VMEM_A
  2. vpack.bf16.fp8 (unit scale) → fp8 A in MRF
  3. Load fp8 B tile (1024 B) → VMEM_B
  4. Push B as MXU weight; run vmatmul / vmatmul.acc
  5. After all K-tiles: vmatpop → (v4, v5) bf16 pair → VMEM_C

DRAM layout (per _make_program):
  [dram_a]  M_tiles × K_tiles × 2048 B  — col-blocked bf16 A
  [dram_b]  K_tiles × N_tiles × 1024 B  — fp8 B tiles (tiled layout)
  [dram_c]  M_tiles × N_tiles × 2048 B  — col-blocked bf16 C output

VMEM slots (fixed, reused per tile):
  0x2000  VMEM_A   2 KB  — bf16 A tile (H0 at 0x2000, H1 at 0x2400)
  0x2800  VMEM_B   1 KB  — fp8 B tile
  0x3000  VMEM_C0  1 KB  — C low  half (cols  0-15)
  0x3400  VMEM_C1  1 KB  — C high half (cols 16-31)

MRF layout per K-tile:
  (v0, v1) = bf16 A halves (LMUL=2 pair)
  v2       = fp8 A (from vpack.bf16.fp8)
  v3       = fp8 B tile
  (v4, v5) = result bf16 pair (vmatpop.bf16.acc.mxu0)

Constraints: M, K, N multiples of 32.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs

VMEM_A = 0x2000    # bf16 A (H0 at 0x2000, H1 at 0x2400 via imm12=32)
VMEM_B = 0x2800    # fp8 B
VMEM_C0 = 0x3000   # C low  half
VMEM_C1 = 0x3400   # C high half

TILE = 32
FP8_BYTES = 1
BF16_BYTES = 2
HALF_BYTES = TILE * (TILE // 2) * BF16_BYTES   # 1024 B (32×16 bf16)
TILE_BYTES_FP8 = TILE * TILE * FP8_BYTES        # 1024 B
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES      # 2048 B


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


def _a_dram_offset(m: int, k: int, M: int, K: int) -> int:
    """Byte offset of the col-blocked bf16 A tile (m, k)."""
    K_tiles = K // TILE
    return (m * K_tiles + k) * TILE_BYTES_BF16


def _b_dram_offset(k: int, n: int, K: int, N: int) -> int:
    """Byte offset of the fp8 B tile (k, n) — tiled layout."""
    N_tiles = N // TILE
    return (k * N_tiles + n) * TILE_BYTES_FP8


def _c_dram_offset(m: int, n: int, M: int, N: int) -> int:
    """Byte offset of the col-blocked bf16 C tile (m, n)."""
    N_tiles = N // TILE
    return (m * N_tiles + n) * TILE_BYTES_BF16


def _colblock_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Tiled col-blocked layout for an M×N bf16 tensor."""
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for r in range(M_tiles):
        for c in range(N_tiles):
            tile = mat[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE]
            parts.append(tile[:, : TILE // 2].contiguous())
            parts.append(tile[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def _tile_matrix_fp8(mat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Flatten fp8 (rows, cols) into tiled row-major order."""
    row_tiles = rows // TILE
    col_tiles = cols // TILE
    parts = []
    for r in range(row_tiles):
        for c in range(col_tiles):
            parts.append(
                mat[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE].contiguous()
            )
    return torch.cat([p.reshape(-1) for p in parts])


def fused_quant_matmul_reference(
    a_bf16: torch.Tensor,
    b_fp8: torch.Tensor,
) -> torch.Tensor:
    """Hardware-faithful reference for fused quantize + matmul.

    Quantizes A to fp8 (unit scale) then accumulates A_fp8 @ B_fp8 per K-tile.
    """
    M, K = a_bf16.shape
    K2, N = b_fp8.shape
    assert K == K2
    K_tiles = K // TILE

    acc = None
    for k in range(K_tiles):
        # Quantize this K-tile of A (unit scale)
        a_q = a_bf16[:, k * TILE : (k + 1) * TILE].to(torch.float8_e4m3fn)
        b_k = b_fp8[k * TILE : (k + 1) * TILE, :]
        prod = a_q.to(torch.float16) @ b_k.to(torch.float16)
        if acc is None:
            acc = prod
        else:
            acc = acc.to(torch.bfloat16).to(torch.float16) + prod
    return acc.to(torch.bfloat16)


def make_fused_quant_matmul_instructions(
    M: int,
    K: int,
    N: int,
    dram_a: int,
    dram_b: int,
    dram_c: int,
) -> list[Instruction]:
    """Generate instructions for fused-quant-matmul M×K×N.

    Scalar register allocation:
        x1  VMEM_A    x2  VMEM_B    x3  VMEM_C0    x4  VMEM_C1
        x5  BF16_TILE_BYTES (2048)  x6  fp8 tile (1024)  x7  HALF_BYTES (1024)
        x10–x12  scratch DRAM addresses
    ERF register 0: unit scale (seli imm=1) set once before main loop.
    """
    assert M % TILE == 0 and K % TILE == 0 and N % TILE == 0
    M_tiles = M // TILE
    K_tiles = K // TILE
    N_tiles = N // TILE

    insns: list[Instruction] = []

    _emit_load_vmem_addr(1, VMEM_A, insns)
    _emit_load_vmem_addr(2, VMEM_B, insns)
    _emit_load_vmem_addr(3, VMEM_C0, insns)
    _emit_load_vmem_addr(4, VMEM_C1, insns)
    _emit_load_imm32(5, TILE_BYTES_BF16, insns)
    _emit_load_imm32(6, TILE_BYTES_FP8, insns)
    _emit_load_imm32(7, HALF_BYTES, insns)

    # Unit scale in ERF[0] once
    insns.append(Instruction("seli", ScalarArgs(rd=0, imm=1)))

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    for m in range(M_tiles):
        for n in range(N_tiles):
            for k in range(K_tiles):
                a_addr = dram_a + _a_dram_offset(m, k, M, K)
                b_addr = dram_b + _b_dram_offset(k, n, K, N)

                # DMA: bf16 A tile (2048 B) → VMEM_A, fp8 B tile (1024 B) → VMEM_B
                _emit_load_imm32(10, a_addr, insns)
                _emit_load_imm32(11, b_addr, insns)
                insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=1, rs1=10, rs2=5, channel=0)))
                insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=11, rs2=6, channel=1)))
                insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
                insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

                # Load LMUL=2 bf16 A pair: (v0, v1)
                insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1, imm12=0)))   # v0 = A H0
                insns.append(Instruction("delay", ScalarArgs(imm=16)))
                insns.append(Instruction("vload", VectorArgs(vd=1, rs1=1, imm12=32)))  # v1 = A H1
                insns.append(Instruction("delay", ScalarArgs(imm=16)))

                # Quantize LMUL=2 pair (v0, v1) → fp8 v2  (unit scale ERF[0]=1)
                insns.append(Instruction("vpack.bf16.fp8", VectorArgs(vd=2, vs1=0, es1=0)))
                insns.append(Instruction("delay", ScalarArgs(imm=66)))

                # Load fp8 B tile → v3
                insns.append(Instruction("vload", VectorArgs(vd=3, rs1=2)))
                insns.append(Instruction("delay", ScalarArgs(imm=16)))

                # Push B weight, then multiply
                insns.append(Instruction("vmatpush.weight.mxu0", VectorArgs(vs1=3)))
                insns.append(Instruction("delay", ScalarArgs(imm=16)))

                if k == 0:
                    insns.append(Instruction("vmatmul.mxu0", MatrixArgs(vs1=2)))
                else:
                    insns.append(Instruction("vmatmul.acc.mxu0", MatrixArgs(vs1=2)))
                insns.append(Instruction("delay", ScalarArgs(imm=32)))

            # All K-tiles done: pop accumulator → (v4, v5) bf16 pair
            insns.append(Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=4)))
            insns.append(Instruction("delay", ScalarArgs(imm=32)))

            # vstore col-blocked halves
            insns.append(Instruction("vstore", VectorArgs(vd=4, rs1=3)))   # H0 → VMEM_C0
            insns.append(Instruction("delay", ScalarArgs(imm=16)))
            insns.append(Instruction("vstore", VectorArgs(vd=5, rs1=4)))   # H1 → VMEM_C1
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

            c_addr = dram_c + _c_dram_offset(m, n, M, N)
            _emit_load_imm32(12, c_addr, insns)
            insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=12, rs1=3, rs2=5, channel=0)))
            insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def _make_program(M: int, K: int, N: int, seed: int):
    dram_a = 0x0000
    dram_b = dram_a + M * K * BF16_BYTES
    dram_c = dram_b + K * N * FP8_BYTES

    torch.manual_seed(seed)
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16) * 0.5
    b_fp8 = torch.randint(-8, 8, (K, N), dtype=torch.int8).float().to(torch.float8_e4m3fn)
    expected = fused_quant_matmul_reference(a_bf16, b_fp8)

    insns = make_fused_quant_matmul_instructions(
        M=M, K=K, N=N, dram_a=dram_a, dram_b=dram_b, dram_c=dram_c
    )
    regions = [
        (dram_a, _colblock_bf16(a_bf16, M, K)),
        (dram_b, _tile_matrix_fp8(b_fp8, K, N)),
    ]
    golden = (dram_c, _colblock_bf16(expected, M, N))
    return insns, regions, golden


# ── 32×32×32: single tile, single K-tile ─────────────────────────────────────

_32_insns, _32_regions, _32_golden = _make_program(32, 32, 32, seed=500)


class ParameterizedFusedQuantMatmul32x32x32Program(Program):
    """Fused quant-matmul 32×32×32 — quantize then 1×1 tile matmul."""

    instructions: List[Instruction[Any]] = _32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _32_regions
    golden_result: tuple[int, torch.Tensor] = _32_golden
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


# ── 32×64×32: single output tile, 2 K-tiles (quantize each) ──────────────────

_kchain_insns, _kchain_regions, _kchain_golden = _make_program(32, 64, 32, seed=501)


class ParameterizedFusedQuantMatmul32x64x32Program(Program):
    """Fused quant-matmul 32×64×32 — quantize 2 K-tiles, K-accumulation."""

    instructions: List[Instruction[Any]] = _kchain_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _kchain_regions
    golden_result: tuple[int, torch.Tensor] = _kchain_golden
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


# ── 64×32×64: 2×2 output tiles, single K-tile ────────────────────────────────

_multi_insns, _multi_regions, _multi_golden = _make_program(64, 32, 64, seed=502)


class ParameterizedFusedQuantMatmul64x32x64Program(Program):
    """Fused quant-matmul 64×32×64 — 2×2 output tiles."""

    instructions: List[Instruction[Any]] = _multi_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _multi_regions
    golden_result: tuple[int, torch.Tensor] = _multi_golden
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)
