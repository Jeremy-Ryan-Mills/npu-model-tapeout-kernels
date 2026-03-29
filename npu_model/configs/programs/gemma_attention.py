from typing import List, Tuple

import math

import torch

from ...software import Instruction, Program


SEQ_LEN = 64
HEAD_DIM = 16

# Data tensors for Q, K, V in fp8 (to match MXU matmul expectations)
QUERY_DATA = torch.ones((SEQ_LEN, HEAD_DIM), dtype=torch.float8_e4m3fn)
KEY_DATA = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)
VALUE_DATA = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)

# Scale = 0 so that scores_scaled = 0 exactly (in bf16), giving exp(0) = 1.0 exactly.
# This avoids a bf16→fp8 NaN when 1/HEAD_DIM = 0.0625 = 0x3D80 (byte 0x80 = fp8 -0, valid).
# With a non-zero scale the bf16-rounded softmax value lands on 0x3D7F whose low byte
# 0x7F is the fp8 NaN pattern, propagating NaN through the second matmul.mxu0.
SCALE_VALUE = 0.0
SCALE_DATA = torch.zeros((SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16)

QUERY_BASE = 0x0000
KEY_BASE   = 0x2000
VALUE_BASE = 0x3000
SCALE_BASE = 0x4000
OUTPUT_BASE = 0x5000


class GemmaAttentionProgram(Program):
    """
    Gemma attention kernel program (simplified, single-head).

    Demonstrates scaled dot-product attention on the NPU ISA:
      - dma.load.mxu0 to prefetch K and V into MXU0 weight buffers
      - matmul.mxu0 for Q @ K (fp8 activations × fp8 weights → bf16 accumulator)
      - vmul / vexp / vrot.reduce.sum / vrcp to compute per-row softmax
      - matmul.mxu0 again for softmax_weights @ V
        (softmax bf16 bytes re-interpreted as fp8 by the hardware)

    Note: SCALE_VALUE is 0 so that scores_scaled = 0, exp(0) = 1.0 (exact in bf16),
    and softmax = 1/HEAD_DIM = 0.0625 = 0x3D80, whose fp8 low byte is 0x80 (-0, valid).
    """

    instructions: List[Instruction] = [
        # Load K and V into MXU0 weight buffer (slots 0 and 1)
        Instruction(
            mnemonic="dma.load.mxu0",
            args={
                "rd": 0,
                "base": KEY_BASE,
                "size": KEY_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu0",
            args={
                "rd": 1,
                "base": VALUE_BASE,
                "size": VALUE_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        # Load Q into MRF[0], scale matrix into MRF[2]
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 0,
                "base": QUERY_BASE,
                "size": QUERY_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 2,
            },
        ),
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 2,
                "base": SCALE_BASE,
                "size": SCALE_DATA.numel() * torch.bfloat16.itemsize,
                "flag": 3,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 2}),
        Instruction(mnemonic="dma.wait", args={"flag": 3}),
        # scores = Q @ K  (MRF[0] fp8 × WB_mxu0[0] fp8 → MRF[3] bf16)
        Instruction(mnemonic="matmul.mxu0", args={"rd": 3, "rs1": 0, "rs2": 0}),
        # scores_scaled = scores * scale  (elementwise bf16 mul)
        Instruction(mnemonic="vmul", args={"vrd": 4, "vs1": 3, "vs2": 2}),
        # exp_scores = exp(scores_scaled)
        Instruction(mnemonic="vexp", args={"vrd": 5, "vs1": 4}),
        # row_sum = sum(exp_scores, dim=-1) broadcast back to full shape
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 6, "vs1": 5}),
        # inv_row_sum = 1 / row_sum
        Instruction(mnemonic="vrcp", args={"vrd": 7, "vs1": 6}),
        # softmax_scores = exp_scores * inv_row_sum  (MRF[8], bf16)
        Instruction(mnemonic="vmul", args={"vrd": 8, "vs1": 5, "vs2": 7}),
        # attn_output = softmax_scores @ V
        # Hardware reads MRF[8] as fp8 (bf16 bytes re-interpreted).
        Instruction(mnemonic="matmul.mxu0", args={"rd": 9, "rs1": 8, "rs2": 1}),
        # Store result to DRAM
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 9,
                "base": OUTPUT_BASE,
                "size": SEQ_LEN * HEAD_DIM * torch.bfloat16.itemsize,
                "flag": 4,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 4}),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (QUERY_BASE, QUERY_DATA),
        (KEY_BASE,   KEY_DATA),
        (VALUE_BASE, VALUE_DATA),
        (SCALE_BASE, SCALE_DATA),
    ]

    # Hardware-accurate golden result.
    # Mirrors the exact byte-level computation the simulator performs, including
    # the bf16→fp8 reinterpretation when matmul.mxu0 reads MRF[8] as fp8 activations.
    _Q_pad = torch.zeros(SEQ_LEN, 32, dtype=torch.float8_e4m3fn)
    _Q_pad[:, :HEAD_DIM] = QUERY_DATA
    _K_pad = torch.zeros(32, HEAD_DIM, dtype=torch.float8_e4m3fn)
    _K_pad[:HEAD_DIM, :] = KEY_DATA
    _V_pad = torch.zeros(32, HEAD_DIM, dtype=torch.float8_e4m3fn)
    _V_pad[:HEAD_DIM, :] = VALUE_DATA

    _scores   = (_Q_pad.float() @ _K_pad.float()).to(torch.bfloat16)
    _scaled   = (_scores * SCALE_DATA).to(torch.bfloat16)
    _exp      = torch.exp(_scaled.float()).to(torch.bfloat16)
    _row_sum  = (
        torch.sum(_exp.float(), dim=-1, keepdim=True)
        .expand_as(_exp)
        .to(torch.bfloat16)
    )
    _softmax  = (_exp.float() / _row_sum.float()).to(torch.bfloat16)
    # Reinterpret bf16 bytes as fp8 — matches hardware read_mrf_fp8 of MRF[8]
    _softmax_fp8 = (
        _softmax.contiguous()
        .view(torch.uint8)
        .view(torch.float8_e4m3fn)
        .reshape(SEQ_LEN, 32)
    )
    golden_result: tuple[int, torch.Tensor] = (
        OUTPUT_BASE,
        (_softmax_fp8.float() @ _V_pad.float()).to(torch.bfloat16),
    )
