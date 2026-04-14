from ...software import m, x, Program
import torch
from ...workload.gemma_blocks import gemma_rms_norm_forward
from npu_model.isa import Instruction

from npu_model.configs.isa_definition import *

# Input shape matches one BF16 tensor register: 32 rows x 16 columns.
INPUT_DATA = torch.randn(32, 16, dtype=torch.bfloat16)
ROW_SIZE = INPUT_DATA.shape[-1]
EPS = 1e-6
# DRAM layout
DRAM_INPUT_BASE = 0x0000
DRAM_EPS_BASE = 0x0400
DRAM_OUTPUT_BASE = 0x0800

# VMEM layout
VMEM_INPUT_BASE = 0x2000
VMEM_EPS_BASE = 0x2400
VMEM_OUTPUT_BASE = 0x2800


class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row-wise mean via transpose + vreduce.sum (second-to-last dim) + vbroadcast.cols.
    """

    instructions: list[Instruction] = [
        # VMEM bases (use LUI+ADDI so immediates stay 12-bit clean)
        # 0x2000
        LUI(rd=x(1), imm=0x2),
        # 0x2400
        LUI(rd=x(2), imm=0x2),
        ADDI(rd=x(2), rs1=x(2), imm=0x400),
        # 0x2800 = 0x3000 - 0x800
        LUI(rd=x(3), imm=0x3),
        ADDI(rd=x(3), rs1=x(3), imm=-2048),
        # DRAM bases
        ADDI(rd=x(4), rs1=x(0), imm=DRAM_INPUT_BASE),
        ADDI(rd=x(5), rs1=x(0), imm=DRAM_EPS_BASE),
        # DRAM_OUTPUT_BASE = 0x0800 = 0x1000 - 0x800
        LUI(rd=x(6), imm=0x1),
        ADDI(rd=x(6), rs1=x(6), imm=-2048),
        # byte length for bf16 tile
        ADDI(rd=x(7), rs1=x(0), imm=1024),

        # DRAM -> VMEM
        DMA_CONFIG_CH0(rs1=x(0)),
        DMA_WAIT_CH0(),
        DMA_LOAD_CH0(rd=x(1), rs1=x(4), rs2=x(7)),
        DMA_LOAD_CH1(rd=x(2), rs1=x(5), rs2=x(7)),
        DMA_WAIT_CH0(),
        DMA_WAIT_CH1(),

        # VMEM -> MRF
        VLOAD(vd=m(0), rs1=x(1), imm=0),  # x
        VLOAD(vd=m(1), rs1=x(2), imm=0),  # eps

        # x_sq = x * x
        VMUL_BF16(vd=m(2), vs1=m(0), vs2=m(0)),
        # sum_sq over columns, broadcast back across each row
        VREDSUM_ROW_BF16(vd=m(3), vs1=m(2)),
        # mean_sq = sum_sq * (1/ROW_SIZE)
        VLI_ALL(vd=m(4), imm=ROW_SIZE),
        VRECIP_BF16(vd=m(5), vs1=m(4)),
        VMUL_BF16(vd=m(6), vs1=m(3), vs2=m(5)),
        # var_eps = var + eps
        VADD_BF16(vd=m(7), vs1=m(6), vs2=m(1)),
        # rsqrt = 1/sqrt(var_eps)
        VSQRT_BF16(vd=m(8), vs1=m(7)),
        VRECIP_BF16(vd=m(9), vs1=m(8)),
        # output = x * rsqrt
        VMUL_BF16(vd=m(10), vs1=m(0), vs2=m(9)),

        # MRF -> VMEM -> DRAM
        VSTORE(vd=m(10), rs1=x(3), imm=0),
        DELAY(imm=20),
        DMA_STORE_CH0(rd=x(6), rs1=x(3), rs2=x(7)),
        DMA_WAIT_CH0(),
    ]

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT_DATA),
        (DRAM_EPS_BASE, torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        gemma_rms_norm_forward(INPUT_DATA).to(torch.bfloat16),
    )
