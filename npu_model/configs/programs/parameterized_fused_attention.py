"""Parameterized fused attention (flash SDPA) kernel.

Constraints (current):
    Q_ROWS  = 32      (one Q block)
    HEAD_DIM = 64     (two 32-wide MXU passes for Q@K^T)
    K_SEQ   = any multiple of 32  (K_TILES = K_SEQ // 32)

Flash attention per K-tile (online softmax):
    m = max(m, rowmax(Q @ K^T * scale))
    exp_diff = exp(m_prev - m)
    O = exp_diff * O + exp(Q @ K^T * scale - m) @ V
    l = exp_diff * l + rowsum(exp(Q @ K^T * scale - m))
    output = O / l

DRAM layouts (all bf16, column-blocked so each vload gets one [32,16] chunk):
    Q:      [32, 64] stored as [128, 16] (4 × [32,16] col blocks)
    KT[k]:  K^T tile k, [64, 32] stored as [128, 16]
    VT[k]:  V_std tile k, [32, 64] stored as [128, 16]
    SCALE:  [32, 16] broadcast value 1/sqrt(HEAD_DIM)
    OUT:    [32, 64] stored as [128, 16]

VMEM layout (fixed, reused across K-tiles):
    VMEM_Q     = 0x0000   4 KB — Q data (loaded once)
    VMEM_KT    = 0x1000   4 KB — current K^T tile (reloaded per K-tile)
    VMEM_VT    = 0x2000   4 KB — current V_std tile (reloaded per K-tile)
    VMEM_SCALE = 0x3000   1 KB — scale broadcast (loaded once)
    VMEM_OUT   = 0x4000   4 KB — output accumulation

MRF register map (bf16 regs are [32,16]; pair-ops use even base + implicit +1):
    Persistent across K-tiles:
        v0,v1   Q_lo bf16 [32,32]  Q[:, 0:32]
        v2,v3   Q_hi bf16 [32,32]  Q[:, 32:64]
        v4      Q_lo fp8  [32,32]
        v5      Q_hi fp8  [32,32]
        v6,v7   scale bf16 (same value in both halves)
        v8,v9   m_prev bf16 [32,16] running rowmax (init -100)
        v10,v11 l_prev bf16 [32,16] running rowsum (init 0)
        v12,v13 O_left  bf16 [32,32] cols 0:32 (init 0)
        v14,v15 O_right bf16 [32,32] cols 32:64 (init 0)
    Per K-tile (overwritten each iteration):
        v16,v17 KT_top bf16   v18,v19 KT_bot bf16
        v20     KT_top fp8    v22     KT_bot fp8
        v24,v25 V_left bf16   v26,v27 V_right bf16
        v28     V_left fp8    v30     V_right fp8
    Temporaries:
        v32,v33 scores bf16    v34,v35 scaled bf16
        v36,v37 tile_max / exp_diff bf16
        v38,v39 m_new bf16
        v40,v41 exp_s bf16     v42     exp_s fp8
        v44,v45 vc_left bf16   v46,v47 vc_right bf16
        v48,v49 scratch        v50,v51 rowsum

Scalar register map:
    x1=VMEM_Q  x2=VMEM_KT  x3=VMEM_VT  x4=VMEM_SCALE  x5=VMEM_OUT
    x6=DRAM_Q  x7=DRAM_SCALE  x8=DRAM_OUT
    x9=tile_bytes(4096)  x10=scale_bytes(1024)
    x11=scratch:DRAM_KT[k]  x12=scratch:DRAM_VT[k]
"""

import math
from typing import Any, List, Tuple

import torch

from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, ScalarArgs, VectorArgs

TILE = 32      # MXU tile edge (fp8 input, bf16 half-output)
BF16 = 2       # bytes per bf16

# Fixed VMEM layout (byte addresses)
VMEM_Q = 0x0000
VMEM_KT = 0x1000
VMEM_VT = 0x2000
VMEM_SCALE = 0x3000
VMEM_OUT = 0x4000


# ── helpers ────────────────────────────────────────────────────────────────

def _emit_load_imm32(rd: int, value: int, out: list) -> None:
    """lui + addi to materialise an arbitrary 32-bit value in rd."""
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


def _col_block(t: torch.Tensor) -> torch.Tensor:
    """Pack [rows, cols] bf16 into column-blocked DRAM layout [N*32, 16].

    Each vload reads one contiguous [32,16] block. Blocks ordered:
    col 0:16 of rows 0:32, col 16:32 of rows 0:32, ..., same for rows 32:64, etc.
    """
    rows, cols = t.shape
    assert rows % TILE == 0 and cols % 16 == 0
    parts = []
    for r in range(0, rows, TILE):
        for c in range(0, cols, 16):
            parts.append(t[r : r + TILE, c : c + 16].contiguous())
    return torch.cat(parts, dim=0)


# ── ISA generator ──────────────────────────────────────────────────────────

def make_fused_attention_instructions(
    Q_ROWS: int,
    K_SEQ: int,
    HEAD_DIM: int,
    dram_q: int,
    dram_kt_base: int,
    dram_vt_base: int,
    dram_scale: int,
    dram_out: int,
) -> list[Instruction]:
    """Generate flash-attention ISA for one Q-block over K_SEQ keys.

    Assumes Q_ROWS=32 and HEAD_DIM=64 (two 32-wide MXU passes for Q@K^T).
    K_SEQ must be a multiple of 32; the loop over K_TILES is unrolled at
    program-generation time.
    """
    assert Q_ROWS == TILE, f"Q_ROWS must be {TILE}, got {Q_ROWS}"
    assert HEAD_DIM == 64, f"HEAD_DIM must be 64 (two MXU passes), got {HEAD_DIM}"
    assert K_SEQ % TILE == 0, f"K_SEQ must be a multiple of {TILE}"

    K_TILES = K_SEQ // TILE
    TILE_BYTES = Q_ROWS * HEAD_DIM * BF16        # 4096 — Q, KT, VT, OUT all same size
    SCALE_BYTES = Q_ROWS * (HEAD_DIM // 4) * BF16  # 1024 — [32, 16] bf16

    insns: list[Instruction] = []

    # ── scalar setup: VMEM addresses ──────────────────────────────────────
    # All VMEM addresses are multiples of 0x1000 → single lui each
    insns.append(Instruction("addi", ScalarArgs(rd=1, rs1=0, imm=0)))     # x1 = VMEM_Q
    insns.append(Instruction("lui",  ScalarArgs(rd=2, imm=VMEM_KT >> 12)))  # x2 = VMEM_KT
    insns.append(Instruction("lui",  ScalarArgs(rd=3, imm=VMEM_VT >> 12)))  # x3 = VMEM_VT
    insns.append(Instruction("lui",  ScalarArgs(rd=4, imm=VMEM_SCALE >> 12)))  # x4 = VMEM_SCALE
    insns.append(Instruction("lui",  ScalarArgs(rd=5, imm=VMEM_OUT >> 12)))    # x5 = VMEM_OUT

    # DRAM addresses and transfer sizes
    _emit_load_imm32(6, dram_q, insns)        # x6 = DRAM_Q
    _emit_load_imm32(7, dram_scale, insns)    # x7 = DRAM_SCALE
    _emit_load_imm32(8, dram_out, insns)      # x8 = DRAM_OUT
    _emit_load_imm32(9, TILE_BYTES, insns)    # x9 = 4096
    insns.append(Instruction("addi", ScalarArgs(rd=10, rs1=0, imm=SCALE_BYTES)))  # x10 = 1024

    # ── DMA: load Q and SCALE once ────────────────────────────────────────
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=0)))
    insns.append(Instruction("dma.config.ch<N>", DmaArgs(channel=1)))
    insns.append(Instruction("dma.wait.ch<N>",   DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>",   DmaArgs(channel=1)))
    insns.append(Instruction("dma.load.ch<N>",   DmaArgs(rd=1, rs1=6, rs2=9,  channel=0)))  # Q
    insns.append(Instruction("dma.load.ch<N>",   DmaArgs(rd=4, rs1=7, rs2=10, channel=1)))  # SCALE
    insns.append(Instruction("dma.wait.ch<N>",   DmaArgs(channel=0)))
    insns.append(Instruction("dma.wait.ch<N>",   DmaArgs(channel=1)))

    # ── Load Q into MRF (4 × [32,16] blocks), quantize bf16 → fp8 ────────
    for i, vd in enumerate([0, 1, 2, 3]):
        insns.append(Instruction("vload", VectorArgs(vd=vd, rs1=1, imm12=i * TILE)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

    # Q_lo (v0, v1) → v4 fp8
    insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=32)))
    insns.append(Instruction("vmatpop.fp8.acc.mxu0",   MatrixArgs(vd=4, vs1=1)))
    insns.append(Instruction("delay", ScalarArgs(imm=16)))

    # Q_hi (v2, v3) → v5 fp8
    insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=2)))
    insns.append(Instruction("delay", ScalarArgs(imm=32)))
    insns.append(Instruction("vmatpop.fp8.acc.mxu0",   MatrixArgs(vd=5, vs1=1)))
    insns.append(Instruction("delay", ScalarArgs(imm=16)))

    # ── Load scale into v6 and v7 (both halves of the pair) ──────────────
    insns.append(Instruction("vload", VectorArgs(vd=6, rs1=4, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=16)))
    insns.append(Instruction("vload", VectorArgs(vd=7, rs1=4, imm12=0)))
    insns.append(Instruction("delay", ScalarArgs(imm=16)))

    # ── Initialize online-softmax state ───────────────────────────────────
    # m_prev (v8, v9) = -100 (proxy for -inf; real logits after scaling >> 100)
    # l_prev (v10, v11) = 0
    # O_left (v12, v13) = 0,  O_right (v14, v15) = 0
    for vd, val in [(8, -100), (9, -100), (10, 0), (11, 0),
                    (12, 0), (13, 0), (14, 0), (15, 0)]:
        insns.append(Instruction("vli.all", VectorArgs(vd=vd, imm=val)))
        insns.append(Instruction("delay",   ScalarArgs(imm=16)))

    # ── K-tile loop (unrolled) ────────────────────────────────────────────
    for k in range(K_TILES):
        kt_addr = dram_kt_base + k * TILE_BYTES
        vt_addr = dram_vt_base + k * TILE_BYTES

        _emit_load_imm32(11, kt_addr, insns)   # x11 = DRAM_KT[k]
        _emit_load_imm32(12, vt_addr, insns)   # x12 = DRAM_VT[k]

        # DMA: load KT and VT in parallel
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=2, rs1=11, rs2=9, channel=0)))
        insns.append(Instruction("dma.load.ch<N>", DmaArgs(rd=3, rs1=12, rs2=9, channel=1)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=0)))
        insns.append(Instruction("dma.wait.ch<N>", DmaArgs(channel=1)))

        # Load KT: (v16,v17) = KT_top, (v18,v19) = KT_bot
        for i, vd in enumerate([16, 17, 18, 19]):
            insns.append(Instruction("vload", VectorArgs(vd=vd, rs1=2, imm12=i * TILE)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # KT_top (v16,v17) → v20 fp8
        insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=16)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.fp8.acc.mxu0",   MatrixArgs(vd=20, vs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # KT_bot (v18,v19) → v22 fp8
        insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=18)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.fp8.acc.mxu0",   MatrixArgs(vd=22, vs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # Load VT: (v24,v25) = V_left, (v26,v27) = V_right
        for i, vd in enumerate([24, 25, 26, 27]):
            insns.append(Instruction("vload", VectorArgs(vd=vd, rs1=3, imm12=i * TILE)))
            insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # V_left (v24,v25) → v28 fp8
        insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=24)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.fp8.acc.mxu0",   MatrixArgs(vd=28, vs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # V_right (v26,v27) → v30 fp8
        insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=26)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.fp8.acc.mxu0",   MatrixArgs(vd=30, vs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # Q @ K^T: acc[0] = Q_lo@KT_top + Q_hi@KT_bot  → (v32,v33) scores
        insns.append(Instruction("vmatpush.weight.mxu0", MatrixArgs(vd=0, vs1=20)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vmatmul.mxu0",         MatrixArgs(vd=0, vs1=4, vs2=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpush.weight.mxu0", MatrixArgs(vd=0, vs1=22)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vmatmul.acc.mxu0",     MatrixArgs(vd=0, vs1=5, vs2=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.bf16.acc.mxu0", MatrixArgs(vd=32, vs1=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))

        # scaled (v34,v35) = scores * scale
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=34, vs1=32, vs2=6)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # tile_max (v36) = rowmax(scaled)
        insns.append(Instruction("vredmax.row.bf16", VectorArgs(vd=36, vs1=34)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # m_new (v38,v39) = max(m_prev, tile_max)
        insns.append(Instruction("vmaximum.bf16", VectorArgs(vd=38, vs1=8, vs2=36)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # exp_diff (v36,v37) = exp(m_prev - m_new)
        insns.append(Instruction("vsub.bf16", VectorArgs(vd=36, vs1=8,  vs2=38)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))
        insns.append(Instruction("vexp.bf16", VectorArgs(vd=36, vs1=36)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # O *= exp_diff
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=12, vs1=12, vs2=36)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))
        insns.append(Instruction("vmul.bf16", VectorArgs(vd=14, vs1=14, vs2=36)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # exp_s (v40,v41) = exp(scaled - m_new)
        insns.append(Instruction("vsub.bf16", VectorArgs(vd=40, vs1=34, vs2=38)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))
        insns.append(Instruction("vexp.bf16", VectorArgs(vd=40, vs1=40)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # quantize exp_s → v42 fp8 via acc roundtrip
        insns.append(Instruction("vmatpush.acc.bf16.mxu0", MatrixArgs(vd=1, vs1=40)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.fp8.acc.mxu0",   MatrixArgs(vd=42, vs1=1)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

        # vc_left (v44,v45) = exp_s @ V_left
        insns.append(Instruction("vmatpush.weight.mxu0",  MatrixArgs(vd=0, vs1=28)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vmatmul.mxu0",           MatrixArgs(vd=0, vs1=42, vs2=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.bf16.acc.mxu0",  MatrixArgs(vd=44, vs1=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))

        # vc_right (v46,v47) = exp_s @ V_right
        insns.append(Instruction("vmatpush.weight.mxu0",  MatrixArgs(vd=0, vs1=30)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vmatmul.mxu0",           MatrixArgs(vd=0, vs1=42, vs2=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))
        insns.append(Instruction("vmatpop.bf16.acc.mxu0",  MatrixArgs(vd=46, vs1=0)))
        insns.append(Instruction("delay", ScalarArgs(imm=32)))

        # O += V contribution
        insns.append(Instruction("vadd.bf16", VectorArgs(vd=12, vs1=12, vs2=44)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))
        insns.append(Instruction("vadd.bf16", VectorArgs(vd=14, vs1=14, vs2=46)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # l = exp_diff*l + rowsum(exp_s)
        insns.append(Instruction("vmul.bf16",      VectorArgs(vd=48, vs1=36, vs2=10)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))
        insns.append(Instruction("vredsum.row.bf16", VectorArgs(vd=50, vs1=40)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))
        insns.append(Instruction("vadd.bf16",      VectorArgs(vd=10, vs1=48, vs2=50)))
        insns.append(Instruction("delay", ScalarArgs(imm=4)))

        # m_prev = m_new (pair copy via two vmov)
        insns.append(Instruction("vmov", VectorArgs(vd=8,  vs1=38)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))
        insns.append(Instruction("vmov", VectorArgs(vd=9,  vs1=39)))
        insns.append(Instruction("delay", ScalarArgs(imm=16)))

    # ── normalize: O /= l ─────────────────────────────────────────────────
    insns.append(Instruction("vrecip.bf16", VectorArgs(vd=48, vs1=10)))
    insns.append(Instruction("delay", ScalarArgs(imm=16)))
    insns.append(Instruction("vmul.bf16",   VectorArgs(vd=12, vs1=12, vs2=48)))
    insns.append(Instruction("delay", ScalarArgs(imm=4)))
    insns.append(Instruction("vmul.bf16",   VectorArgs(vd=14, vs1=14, vs2=48)))
    insns.append(Instruction("delay", ScalarArgs(imm=4)))

    # ── store output (v12..v15 → VMEM_OUT → DRAM_OUT) ────────────────────
    for i, vd in enumerate([12, 13, 14, 15]):
        insns.append(Instruction("vstore", VectorArgs(vd=vd, rs1=5, imm12=i * TILE)))
        insns.append(Instruction("delay",  ScalarArgs(imm=16)))
    insns.append(Instruction("dma.store.ch<N>", DmaArgs(rd=8, rs1=5, rs2=9, channel=0)))
    insns.append(Instruction("dma.wait.ch<N>",  DmaArgs(channel=0)))
    insns.append(Instruction("ecall", ScalarArgs()))

    return insns


# ── ISA-exact golden reference ─────────────────────────────────────────────

def fused_attention_golden(
    Q_raw: torch.Tensor,       # [Q_ROWS, HEAD_DIM] bf16
    Ks_raw: list,              # K_TILES × [TILE, HEAD_DIM] bf16
    Vs_mlir: list,             # K_TILES × [HEAD_DIM, TILE] bf16  (MLIR layout)
    scale_val: float,
) -> torch.Tensor:
    """Simulate ISA operations exactly, returning [128, 16] bf16 (col-blocked output)."""
    Q_ROWS, HEAD_DIM = Q_raw.shape
    H4 = HEAD_DIM // 4  # = 16

    Q_lo_fp8 = Q_raw[:, :TILE].to(torch.float8_e4m3fn)
    Q_hi_fp8 = Q_raw[:, TILE:].to(torch.float8_e4m3fn)

    scale_r = torch.full((Q_ROWS, H4), scale_val, dtype=torch.bfloat16)
    m   = torch.full((Q_ROWS, H4), -100.0, dtype=torch.bfloat16)
    l   = torch.zeros(Q_ROWS, H4, dtype=torch.bfloat16)
    Oll = torch.zeros(Q_ROWS, H4, dtype=torch.bfloat16)  # O[:, 0:16]
    Olr = torch.zeros(Q_ROWS, H4, dtype=torch.bfloat16)  # O[:, 16:32]
    Orl = torch.zeros(Q_ROWS, H4, dtype=torch.bfloat16)  # O[:, 32:48]
    Orr = torch.zeros(Q_ROWS, H4, dtype=torch.bfloat16)  # O[:, 48:64]

    for k_raw, v_mlir in zip(Ks_raw, Vs_mlir):
        KT = k_raw.T.contiguous()                         # [HEAD_DIM, TILE]
        KT_top_fp8 = KT[:TILE, :].to(torch.float8_e4m3fn)
        KT_bot_fp8 = KT[TILE:, :].to(torch.float8_e4m3fn)

        V_std = v_mlir.T.contiguous()                     # [TILE, HEAD_DIM]
        V_left_fp8  = V_std[:, :TILE].to(torch.float8_e4m3fn)
        V_right_fp8 = V_std[:, TILE:].to(torch.float8_e4m3fn)

        # Q @ K^T (two MXU passes, bf16 accumulation)
        scores = (
            Q_lo_fp8.to(torch.float16) @ KT_top_fp8.to(torch.float16)
            + Q_hi_fp8.to(torch.float16) @ KT_bot_fp8.to(torch.float16)
        ).to(torch.bfloat16)  # [32, 32]

        sl = (scores[:, :H4*1] * scale_r).to(torch.bfloat16)  # left  half [32,16]
        sr = (scores[:, H4*1:] * scale_r).to(torch.bfloat16)  # right half [32,16]

        rm_l = sl.max(dim=1, keepdim=True).values.expand(-1, H4).to(torch.bfloat16)
        rm_r = sr.max(dim=1, keepdim=True).values.expand(-1, H4).to(torch.bfloat16)
        tile_max = torch.maximum(rm_l, rm_r).to(torch.bfloat16)
        m_new = torch.maximum(m, tile_max).to(torch.bfloat16)

        exp_diff = torch.exp((m - m_new).to(torch.bfloat16)).to(torch.bfloat16)

        Oll = (Oll * exp_diff).to(torch.bfloat16)
        Olr = (Olr * exp_diff).to(torch.bfloat16)
        Orl = (Orl * exp_diff).to(torch.bfloat16)
        Orr = (Orr * exp_diff).to(torch.bfloat16)

        esl = torch.exp((sl - m_new).to(torch.bfloat16)).to(torch.bfloat16)
        esr = torch.exp((sr - m_new).to(torch.bfloat16)).to(torch.bfloat16)

        exp_s_fp8 = torch.cat([esl, esr], dim=1).to(torch.float8_e4m3fn)

        vc_left  = (exp_s_fp8.to(torch.float16) @ V_left_fp8.to(torch.float16)).to(torch.bfloat16)
        vc_right = (exp_s_fp8.to(torch.float16) @ V_right_fp8.to(torch.float16)).to(torch.bfloat16)

        Oll = (Oll + vc_left[:, :H4]).to(torch.bfloat16)
        Olr = (Olr + vc_left[:, H4:]).to(torch.bfloat16)
        Orl = (Orl + vc_right[:, :H4]).to(torch.bfloat16)
        Orr = (Orr + vc_right[:, H4:]).to(torch.bfloat16)

        rs_l = esl.sum(dim=1, keepdim=True).expand(-1, H4).to(torch.bfloat16)
        rs_r = esr.sum(dim=1, keepdim=True).expand(-1, H4).to(torch.bfloat16)
        l = ((exp_diff * l).to(torch.bfloat16) + (rs_l + rs_r).to(torch.bfloat16)).to(torch.bfloat16)
        m = m_new

    inv_l = (1.0 / l).to(torch.bfloat16)
    Oll = (Oll * inv_l).to(torch.bfloat16)
    Olr = (Olr * inv_l).to(torch.bfloat16)
    Orl = (Orl * inv_l).to(torch.bfloat16)
    Orr = (Orr * inv_l).to(torch.bfloat16)

    # Column-blocked output: [128, 16] — matches vstore order (v12,v13,v14,v15)
    return torch.cat([Oll, Olr, Orl, Orr], dim=0)


def _make_attn_program(K_SEQ: int, seed: int):
    """Build (instructions, memory_regions, golden_result) for K_SEQ keys."""
    Q_ROWS  = TILE      # 32
    HEAD_DIM = 64
    K_TILES = K_SEQ // TILE

    TILE_BYTES  = Q_ROWS * HEAD_DIM * BF16    # 4096
    SCALE_BYTES = Q_ROWS * (HEAD_DIM // 4) * BF16  # 1024

    dram_q      = 0x0000
    dram_kt_base = dram_q + TILE_BYTES                           # 0x1000
    dram_vt_base = dram_kt_base + K_TILES * TILE_BYTES
    dram_scale   = dram_vt_base + K_TILES * TILE_BYTES
    dram_out     = dram_scale + TILE_BYTES  # generous alignment

    torch.manual_seed(seed)
    Q_raw  = (torch.randn(Q_ROWS, HEAD_DIM) * 0.5).to(torch.bfloat16)
    Ks_raw = [(torch.randn(TILE, HEAD_DIM) * 0.5).to(torch.bfloat16) for _ in range(K_TILES)]
    Vs_mlir = [(torch.randn(HEAD_DIM, TILE) * 0.5).to(torch.bfloat16) for _ in range(K_TILES)]

    scale_val = 1.0 / math.sqrt(float(HEAD_DIM))
    scale_data = torch.full((Q_ROWS, HEAD_DIM // 4), scale_val, dtype=torch.bfloat16)

    # Build DRAM memory regions
    regions = [(dram_q, _col_block(Q_raw))]
    for k, (k_raw, v_mlir) in enumerate(zip(Ks_raw, Vs_mlir)):
        regions.append((dram_kt_base + k * TILE_BYTES, _col_block(k_raw.T.contiguous())))
        regions.append((dram_vt_base + k * TILE_BYTES, _col_block(v_mlir.T.contiguous())))
    regions.append((dram_scale, scale_data))

    expected = fused_attention_golden(Q_raw, Ks_raw, Vs_mlir, scale_val)

    insns = make_fused_attention_instructions(
        Q_ROWS=Q_ROWS, K_SEQ=K_SEQ, HEAD_DIM=HEAD_DIM,
        dram_q=dram_q, dram_kt_base=dram_kt_base,
        dram_vt_base=dram_vt_base, dram_scale=dram_scale, dram_out=dram_out,
    )
    return insns, regions, (dram_out, expected)


# ── K_SEQ=64 (2 K-tiles) — matches SmolVLAFusedAttentionProgram shape ────────

_fa64_insns, _fa64_regions, _fa64_golden = _make_attn_program(K_SEQ=64,  seed=10)


class ParameterizedFusedAttention64Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=64, HEAD_DIM=64 (2 K-tiles).

    Matches the shape used in SmolVLAFusedAttentionProgram but uses a
    generated instruction sequence rather than a hand-written one.
    """

    instructions: List[Instruction[Any]] = _fa64_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa64_regions
    golden_result: tuple[int, torch.Tensor] = _fa64_golden
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)


# ── K_SEQ=96 (3 K-tiles) — tests the loop over more than 2 K-tiles ───────────

_fa96_insns, _fa96_regions, _fa96_golden = _make_attn_program(K_SEQ=96,  seed=11)


class ParameterizedFusedAttention96Program(Program):
    """Flash attention: Q_ROWS=32, K_SEQ=96, HEAD_DIM=64 (3 K-tiles).

    Exercises the K-tile loop with an odd number of tiles (3), verifying
    that the online softmax accumulation works beyond the 2-tile case.
    """

    instructions: List[Instruction[Any]] = _fa96_insns
    memory_regions: List[Tuple[int, torch.Tensor]] = _fa96_regions
    golden_result: tuple[int, torch.Tensor] = _fa96_golden
    kernel_tolerance: tuple[float, float] = (5e-2, 5e-2)
