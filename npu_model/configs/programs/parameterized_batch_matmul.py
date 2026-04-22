"""Parameterized batch matmul: C[b] = A[b] @ B[b] for B batches of M×K×N fp8.

Each batch element is an independent fp8 tiled matmul.  The batch
dimension is unrolled at program-generation time — the hardware has no
loop instructions, so each batch tile is emitted as a separate sequence
of DMA + MXU instructions.

DRAM layout (per _make_program):
  [dram_a]  B × M_tiles × K_tiles × 1024 B  — fp8 A tiles, batch-major
  [dram_b]  B × K_tiles × N_tiles × 1024 B  — fp8 B tiles, batch-major
  [dram_c]  B × M_tiles × N_tiles × 2048 B  — col-blocked bf16 C tiles

VMEM slots (fixed, reused per tile):
  0x2000  VMEM_A   1 KB — current fp8 A tile
  0x2400  VMEM_B   1 KB — current fp8 B tile
  0x2800  VMEM_C0  1 KB — C low  half (cols  0-15)
  0x2C00  VMEM_C1  1 KB — C high half (cols 16-31)

Constraints:
    - B, M, K, N must satisfy M, K, N % 32 == 0.
"""

from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs

VMEM_A = 0x2000
VMEM_B = 0x2400
VMEM_C0 = 0x2800
VMEM_C1 = 0x2C00

TILE = 32
FP8_BYTES = 1
BF16_BYTES = 2
TILE_BYTES_FP8 = TILE * TILE * FP8_BYTES    # 1024 B
TILE_BYTES_BF16 = TILE * TILE * BF16_BYTES  # 2048 B (two col-blocked halves)


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


def _a_tile_offset(b: int, m: int, k: int, M: int, K: int) -> int:
    M_tiles = M // TILE
    K_tiles = K // TILE
    return (b * M_tiles * K_tiles + m * K_tiles + k) * TILE_BYTES_FP8


def _b_tile_offset(b: int, k: int, n: int, K: int, N: int) -> int:
    K_tiles = K // TILE
    N_tiles = N // TILE
    return (b * K_tiles * N_tiles + k * N_tiles + n) * TILE_BYTES_FP8


def _c_tile_offset(b: int, m: int, n: int, M: int, N: int) -> int:
    M_tiles = M // TILE
    N_tiles = N // TILE
    return (b * M_tiles * N_tiles + m * N_tiles + n) * TILE_BYTES_BF16


def _tile_matrix(mat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Flatten (rows, cols) into tiled row-major tile order (each tile contiguous)."""
    row_tiles = rows // TILE
    col_tiles = cols // TILE
    parts = []
    for r in range(row_tiles):
        for c in range(col_tiles):
            parts.append(
                mat[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE].contiguous()
            )
    return torch.cat([p.reshape(-1) for p in parts])


def _colblock_bf16(mat: torch.Tensor, M: int, N: int) -> torch.Tensor:
    """Col-blocked layout: for each tile, H0 (cols 0-15) then H1 (cols 16-31)."""
    M_tiles = M // TILE
    N_tiles = N // TILE
    parts = []
    for r in range(M_tiles):
        for c in range(N_tiles):
            tile = mat[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE]
            parts.append(tile[:, : TILE // 2].contiguous())
            parts.append(tile[:, TILE // 2 :].contiguous())
    return torch.cat(parts, dim=0)


def batch_matmul_reference(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Hardware-faithful batch matmul: fp8 inputs, bf16 accumulation per K-tile.

    a: (B, M, K) fp8 — activations
    b: (B, K, N) fp8 — weights
    Returns: (B, M, N) bf16
    """
    B, M, K = a.shape
    N = b.shape[2]
    K_tiles = K // TILE
    results = []
    for bi in range(B):
        acc = None
        for k in range(K_tiles):
            a_k = a[bi, :, k * TILE : (k + 1) * TILE].to(torch.float16)
            b_k = b[bi, k * TILE : (k + 1) * TILE, :].to(torch.float16)
            if acc is None:
                acc = a_k @ b_k
            else:
                acc = acc.to(torch.bfloat16).to(torch.float16) + (a_k @ b_k)
        results.append(acc.to(torch.bfloat16))
    return torch.stack(results)


def make_batch_matmul_instructions(
    B: int,
    M: int,
    K: int,
    N: int,
    dram_a: int,
    dram_b: int,
    dram_c: int,
) -> list[Instruction]:
    """Generate instructions for a B-batched M×K×N fp8 tiled matmul.

    Scalar register allocation:
        x1  VMEM_A   x2  VMEM_B   x3  VMEM_C0   x4  VMEM_C1
        x5  fp8 tile size (1024 B)  x6  bf16 tile size (2048 B)
        x10–x12  scratch DRAM addresses
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
    insns.append(Instruction("addi", ScalarArgs(rd=5, rs1=0, imm=TILE_BYTES_FP8)))
    _emit_load_imm32(6, TILE_BYTES_BF16, insns)

    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

    for bi in range(B):
        for m in range(M_tiles):
            for n in range(N_tiles):
                for k in range(K_tiles):
                    a_addr = dram_a + _a_tile_offset(bi, m, k, M, K)
                    b_addr = dram_b + _b_tile_offset(bi, k, n, K, N)

                    _emit_load_imm32(10, a_addr, insns)
                    _emit_load_imm32(11, b_addr, insns)

                    insns.append(Instruction(
                        "dma.load.ch<N>", DmaArgs(rd=1, rs1=10, rs2=5, channel=0)
                    ))
                    insns.append(Instruction(
                        "dma.load.ch<N>", DmaArgs(rd=2, rs1=11, rs2=5, channel=1)
                    ))
                    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
                    insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

                    insns.append(Instruction("vload", VectorArgs(vd=0, rs1=1)))  # A tile
                    insns.append(Instruction("delay", ScalarArgs(imm=16)))
                    insns.append(Instruction("vload", VectorArgs(vd=1, rs1=2)))  # B tile
                    insns.append(Instruction("delay", ScalarArgs(imm=16)))

                    insns.append(Instruction("vmatpush.weight.mxu0", VectorArgs(vs1=1)))
                    insns.append(Instruction("delay", ScalarArgs(imm=16)))

                    if k == 0:
                        insns.append(Instruction("vmatmul.mxu0", MatrixArgs(vs1=0)))
                    else:
                        insns.append(Instruction("vmatmul.acc.mxu0", MatrixArgs(vs1=0)))
                    insns.append(Instruction("delay", ScalarArgs(imm=32)))

                # All K-tiles done: pop accumulator → (v2, v3) bf16 pair
                insns.append(Instruction("vmatpop.bf16.acc.mxu0", VectorArgs(vd=2)))
                insns.append(Instruction("delay", ScalarArgs(imm=32)))

                # vstore col-blocked halves to VMEM_C0 / VMEM_C1
                insns.append(Instruction("vstore", VectorArgs(vd=2, rs1=3)))  # low  half
                insns.append(Instruction("delay", ScalarArgs(imm=16)))
                insns.append(Instruction("vstore", VectorArgs(vd=3, rs1=4)))  # high half
                insns.append(Instruction("delay", ScalarArgs(imm=16)))

                c_addr = dram_c + _c_tile_offset(bi, m, n, M, N)
                _emit_load_imm32(12, c_addr, insns)
                insns.append(Instruction(
                    "dma.store.ch<N>", DmaArgs(rd=12, rs1=3, rs2=6, channel=0)
                ))
                insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))

    insns.append(Instruction("ecall", ScalarArgs()))
    return insns


def _batch_expected_stacked(result: torch.Tensor, B: int, M: int, N: int) -> torch.Tensor:
    """Reorder (B, M, N) expected into the flat tiled col-blocked DRAM layout."""
    parts = []
    for bi in range(B):
        parts.append(_colblock_bf16(result[bi], M, N))
    return torch.cat(parts, dim=0)


def _make_program(B: int, M: int, K: int, N: int, seed: int):
    dram_a = 0x0000
    dram_b = dram_a + B * M * K * FP8_BYTES
    dram_c = dram_b + B * K * N * FP8_BYTES

    torch.manual_seed(seed)
    a = torch.randint(-8, 8, (B, M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    b = torch.randint(-8, 8, (B, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)
    expected = batch_matmul_reference(a, b)

    # Tile each batch element's A and B
    a_tiled_parts = [_tile_matrix(a[bi], M, K) for bi in range(B)]
    b_tiled_parts = [_tile_matrix(b[bi], K, N) for bi in range(B)]
    a_dram = torch.cat(a_tiled_parts)
    b_dram = torch.cat(b_tiled_parts)

    insns = make_batch_matmul_instructions(
        B=B, M=M, K=K, N=N,
        dram_a=dram_a, dram_b=dram_b, dram_c=dram_c,
    )
    regions = [(dram_a, a_dram), (dram_b, b_dram)]
    golden = (dram_c, _batch_expected_stacked(expected, B, M, N))
    return insns, regions, golden


# ── 2×32×32×32: 2 batches, 1×1 output tiles, 1 K-tile each ──────────────────

_2x32_insns, _2x32_regions, _2x32_golden = _make_program(2, 32, 32, 32, seed=300)


class ParameterizedBatchMatmul2x32x32x32Program(Program):
    """Batch matmul (B=2, M=K=N=32): 2 independent 32×32×32 fp8 matmuls."""

    instructions: List[Instruction[Any]] = _2x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _2x32_regions
    golden_result: tuple[int, torch.Tensor] = _2x32_golden


# ── 4×32×64×32: 4 batches, 1×1 output tiles, 2 K-tiles each ─────────────────

_4x32_insns, _4x32_regions, _4x32_golden = _make_program(4, 32, 64, 32, seed=301)


class ParameterizedBatchMatmul4x32x64x32Program(Program):
    """Batch matmul (B=4, M=32, K=64, N=32): 4 batches, K-accumulation path."""

    instructions: List[Instruction[Any]] = _4x32_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _4x32_regions
    golden_result: tuple[int, torch.Tensor] = _4x32_golden


# ── 2×64×32×64: 2 batches, 2×2 output tiles, 1 K-tile each ──────────────────

_2x64_insns, _2x64_regions, _2x64_golden = _make_program(2, 64, 32, 64, seed=302)


class ParameterizedBatchMatmul2x64x32x64Program(Program):
    """Batch matmul (B=2, M=64, K=32, N=64): 2 batches, multi-tile output."""

    instructions: List[Instruction[Any]] = _2x64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _2x64_regions
    golden_result: tuple[int, torch.Tensor] = _2x64_golden
