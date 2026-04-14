from ...software import a, m, x, w, Program
import torch
from npu_model.isa import Instruction
from npu_model.workload.gemma_blocks import gemma_mlp_gate_up_forward

from ..isa_definition import *

GATE_PROJ_WEIGHT_DATA = torch.ones((32, 32), dtype=torch.float8_e4m3fn)
UP_PROJ_WEIGHT_DATA = torch.ones((32, 32), dtype=torch.float8_e4m3fn)
ACTIVATION_DATA = torch.ones((32, 32), dtype=torch.float8_e4m3fn)

# DRAM memory layout (program-loaded)
DRAM_GATE_WEIGHT_BASE = 0x0000
DRAM_UP_WEIGHT_BASE = 0x0400  # 1024 bytes after gate
DRAM_ACTIVATION_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x0C00

# VMEM scratchpad layout
VMEM_GATE_WEIGHT_BASE = 0x2000
VMEM_UP_WEIGHT_BASE = 0x2400
VMEM_ACTIVATION_BASE = 0x2800
VMEM_OUTPUT_BASE = 0x2C00


class GemmaMlpProgram(Program):
    """
    Gemma MLP kernel program (simplified).
    Gate and up projections, then elementwise gate*up (simplified GeGLU).
    """

    instructions: list[Instruction] = [
        # x1..x4: VMEM bases (use LUI+ADDI so immediates stay 12-bit clean)
        # 0x2000
        LUI(rd=x(1), imm=0x2),
        # 0x2400
        LUI(rd=x(2), imm=0x2),
        ADDI(rd=x(2), rs1=x(2), imm=0x400),
        # 0x2800 = 0x3000 - 0x800
        LUI(rd=x(3), imm=0x3),
        ADDI(rd=x(3), rs1=x(3), imm=-2048),
        # 0x2C00 = 0x3000 - 0x400
        LUI(rd=x(4), imm=0x3),
        ADDI(rd=x(4), rs1=x(4), imm=-1024),
        # x5..x8: DRAM bases
        ADDI(rd=x(5), rs1=x(0), imm=DRAM_GATE_WEIGHT_BASE),
        ADDI(rd=x(6), rs1=x(0), imm=DRAM_UP_WEIGHT_BASE),
        # DRAM_ACTIVATION_BASE = 0x0800 = 0x1000 - 0x800
        LUI(rd=x(7), imm=0x1),
        ADDI(rd=x(7), rs1=x(7), imm=-2048),
        # DRAM_OUTPUT_BASE = 0x0C00 does not fit in signed 12-bit addi; use LUI/ADDI.
        LUI(rd=x(8), imm=0x1),
        ADDI(rd=x(8), rs1=x(8), imm=-1024),
        # x9: byte length for fp8 tile (32*32*1 = 1024)
        ADDI(rd=x(9), rs1=x(0), imm=1024),
        # x10: byte length for bf16 tile (32*16*2 = 1024)
        ADDI(rd=x(10), rs1=x(0), imm=1024),

        # DRAM -> VMEM
        DMA_CONFIG_CH0(rs1=x(0)),
        DMA_WAIT_CH0(),
        DMA_LOAD_CH0(rd=x(1), rs1=x(5), rs2=x(9)),
        DMA_LOAD_CH1(rd=x(2), rs1=x(6), rs2=x(9)),
        DMA_LOAD_CH2(rd=x(3), rs1=x(7), rs2=x(9)),
        DMA_WAIT_CH0(),
        DMA_WAIT_CH1(),
        DMA_WAIT_CH2(),

        # VMEM -> MRF (weights + activation)
        VLOAD(vd=m(0), rs1=x(1), imm=0),  # gate W (fp8
        VLOAD(vd=m(1), rs1=x(2), imm=0),  # up W (fp8
        VLOAD(vd=m(2), rs1=x(3), imm=0),  # act (fp8

        # Push weights to MXU0 WB slots 0 and 1
        VMATPUSH_WEIGHT_MXU0(vd=w(0), vs1=m(0)),
        VMATPUSH_WEIGHT_MXU0(vd=w(1), vs1=m(1)),
        DELAY(imm=17),

        # --- PHASE 3: Matrix Multiplications ---
        # Gate projection: activation @ gate_weight -> Acc/MRF
        # Note: Using MatrixArgs for matmul
        VMATMUL_MXU0(vd=a(0), vs1=m(2), vs2=w(0)),
        DELAY(imm=33),
        VMATPOP_BF16_ACC_MXU0(vd=m(4), vs2=a(0)),  # gate -> mrf4+5
        # Up projection: activation @ up_weight -> Acc/MRF
        VMATMUL_MXU0(vd=a(0), vs1=m(2), vs2=w(1)),
        DELAY(imm=33),
        VMATPOP_BF16_ACC_MXU0(vd=m(6), vs2=a(0)),  # up -> mrf6+7
        # --- PHASE 4: Element-wise Multiplication (GeGLU Simplified) ---
        VMUL_BF16(vd=m(8), vs1=m(4), vs2=m(6)),
        VMUL_BF16(vd=m(9), vs1=m(5), vs2=m(7)),
        # --- PHASE 5: Store Results ---
        VSTORE(vd=m(8), rs1=x(4), imm=0),
        VSTORE(vd=m(9), rs1=x(4), imm=32),
        DELAY(imm=40),

        # VMEM -> DRAM (two 1024B tiles)
        ADDI(rd=x(11), rs1=x(4), imm=1024),  # vmem+1024
        ADDI(rd=x(12), rs1=x(8), imm=1024),  # dram+1024
        DMA_STORE_CH0(rd=x(8), rs1=x(4), rs2=x(10)),
        DMA_STORE_CH1(rd=x(12), rs1=x(11), rs2=x(10)),
        DMA_WAIT_CH0(),
        DMA_WAIT_CH1(),
    ]

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_GATE_WEIGHT_BASE, GATE_PROJ_WEIGHT_DATA),
        (DRAM_UP_WEIGHT_BASE, UP_PROJ_WEIGHT_DATA),
        (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        # Pure torch reference for the math, then expressed in the same memory layout
        # the program stores: two 32x16 bf16 tiles back-to-back in DRAM.
        torch.cat(
            (
                gemma_mlp_gate_up_forward(
                    ACTIVATION_DATA,
                    GATE_PROJ_WEIGHT_DATA,
                    UP_PROJ_WEIGHT_DATA,
                    use_gelu=False,  # matches NPU: gate * up
                )
                .to(torch.bfloat16)[:, :16],
                gemma_mlp_gate_up_forward(
                    ACTIVATION_DATA,
                    GATE_PROJ_WEIGHT_DATA,
                    UP_PROJ_WEIGHT_DATA,
                    use_gelu=False,  # matches NPU: gate * up
                )
                .to(torch.bfloat16)[:, 16:],
            ),
            dim=0,
        ).contiguous(),
    )
