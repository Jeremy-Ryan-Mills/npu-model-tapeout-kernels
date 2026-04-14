import math
import torch
from ...software import a, m, x, w, Program
from npu_model.isa import Instruction

from npu_model.configs.isa_definition import *

# NOTE: This program is currently written for a single 32x16 tile.
# Use SEQ_LEN=32 so Q @ K produces one bf16 tile (32x16) in this model.
SEQ_LEN = 32
HEAD_DIM = 16

# Data tensors in fp8 sized to one 32x32 tile (what MXU expects).
# We encode logical (SEQ_LEN x HEAD_DIM) by zero-padding within the tile:
# - Q: ones in first 16 cols, zeros elsewhere
# - K: ones in first 16 rows/cols block, zeros elsewhere
QUERY_DATA = torch.zeros((32, 32), dtype=torch.float8_e4m3fn)
QUERY_DATA[:, :HEAD_DIM] = torch.ones((32, HEAD_DIM), dtype=torch.float8_e4m3fn)
KEY_DATA = torch.zeros((32, 32), dtype=torch.float8_e4m3fn)
KEY_DATA[:HEAD_DIM, :HEAD_DIM] = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)

# Scaling matrix: every entry is 1 / sqrt(HEAD_DIM), in bf16 for vector ops
SCALE_VALUE = 1.0 / math.sqrt(float(HEAD_DIM))
SCALE_DATA = torch.full((SEQ_LEN, HEAD_DIM), SCALE_VALUE, dtype=torch.bfloat16)

# DRAM layout (program-loaded)
DRAM_QUERY_BASE = 0x0000
DRAM_KEY_BASE = 0x0400
DRAM_SCALE_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x0C00

# VMEM layout
VMEM_QUERY_BASE = 0x2000
VMEM_KEY_BASE = 0x2400
VMEM_SCALE_BASE = 0x2800
VMEM_OUTPUT_BASE = 0x2C00


class GemmaAttentionProgram(Program):
    """
    Gemma attention kernel program (simplified, single-head).

    This program demonstrates a scaled dot-product attention block using
    the NPU ISA:
      - `matmul.mxu0` for Q @ K and softmax(QK^T) @ V
      - `vexp`, `vreduce.sum`, `vrcp`, and `vmul` to implement softmax.
    """

    instructions: list[Instruction] = [
        # Register setup (VMEM) (use LUI+ADDI so immediates stay 12-bit clean)
        # 0x2000
        LUI(rd=x(1), imm=0x2),
        # 0x2400
        LUI(rd=x(2), imm=0x2),
        ADDI(rd=x(2), rs1=x(2), imm=0x400),
        # 0x2800 = 0x3000 - 0x800
        LUI(rd=x(4), imm=0x3),
        ADDI(rd=x(4), rs1=x(4), imm=-2048),
        # 0x2C00 = 0x3000 - 0x400
        LUI(rd=x(5), imm=0x3),
        ADDI(rd=x(5), rs1=x(5), imm=-1024),
        # Register setup (DRAM)
        ADDI(rd=x(6), rs1=x(0), imm=DRAM_QUERY_BASE),
        ADDI(rd=x(7), rs1=x(0), imm=DRAM_KEY_BASE),
        # DRAM_SCALE_BASE = 0x0800 = 0x1000 - 0x800
        LUI(rd=x(9), imm=0x1),
        ADDI(rd=x(9), rs1=x(9), imm=-2048),
        # DRAM_OUTPUT_BASE = 0x0C00 = 0x1000 - 0x400
        LUI(rd=x(10), imm=0x1),
        ADDI(rd=x(10), rs1=x(10), imm=-1024),
        # Byte lengths: fp8 tile (1024) and bf16 tile (1024)
        ADDI(rd=x(11), rs1=x(0), imm=1024),
        ADDI(rd=x(12), rs1=x(0), imm=1024),

        # DRAM -> VMEM
        DMA_CONFIG_CH0(rs1=x(0)),
        DMA_WAIT_CH0(),
        DMA_LOAD_CH0(rd=x(1), rs1=x(6), rs2=x(11)),  # Q tile
        DMA_LOAD_CH1(rd=x(2), rs1=x(7), rs2=x(11)),  # K tile
        DMA_LOAD_CH2(rd=x(4), rs1=x(9), rs2=x(12)),  # scale (bf16 tile
        DMA_WAIT_CH0(),
        DMA_WAIT_CH1(),
        DMA_WAIT_CH2(),

        # VMEM -> MRF
        VLOAD(vd=m(0), rs1=x(1), imm=0),  # Q (fp8 tile
        VLOAD(vd=m(1), rs1=x(2), imm=0),  # K (fp8 tile
        VLOAD(vd=m(2), rs1=x(4), imm=0),  # scale (bf16 tile, 32x16

        # Push K to WB slot 0, compute scores = Q @ K, pop bf16
        VMATPUSH_WEIGHT_MXU0(vd=w(0), vs1=m(1)),
        DELAY(imm=17),
        VMATMUL_MXU0(vd=a(0), vs1=m(0), vs2=w(0)),
        DELAY(imm=33),
        VMATPOP_BF16_ACC_MXU0(vd=m(3), vs2=a(0)),  # scores bf16 tile

        # scores_scaled = scores * scale
        VMUL_BF16(vd=m(4), vs1=m(3), vs2=m(2)),

        # Softmax (unnormalized variant: no max subtraction)
        # exp_scores = exp(scores_scaled)
        VEXP_BF16(vd=m(5), vs1=m(4)),
        # row_sum = sum(exp_scores) broadcast across columns
        VREDSUM_ROW_BF16(vd=m(6), vs1=m(5)),
        # inv_row_sum = 1 / row_sum
        VRECIP_BF16(vd=m(7), vs1=m(6)),
        # softmax_scores = exp_scores * inv_row_sum
        VMUL_BF16(vd=m(8), vs1=m(5), vs2=m(7)),

        # Store softmax scores (bf16 tile)
        VSTORE(vd=m(8), rs1=x(5), imm=0),
        DELAY(imm=20),
        DMA_STORE_CH0(rd=x(10), rs1=x(5), rs2=x(12)),
        DMA_WAIT_CH0(),
    ]

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_QUERY_BASE, QUERY_DATA),
        (DRAM_KEY_BASE, KEY_DATA),
        (DRAM_SCALE_BASE, SCALE_DATA),
    ]

    # Golden result: softmax(scores_scaled) (no max subtraction), pure torch
    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        torch.softmax(
            ((QUERY_DATA.to(torch.float32) @ KEY_DATA.to(torch.float32)) * SCALE_VALUE)[
                :, :HEAD_DIM
            ],
            dim=1,
        ).to(torch.bfloat16),
    )
