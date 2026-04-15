import torch
from npu_model.isa import Instruction
from ...software import acc, m, x, w, Program

from npu_model.configs.isa_definition import *

# Constants for memory layout
DRAM_ACTIVATION_BASE = 0x0000
DRAM_WEIGHT_BASE = 0x0400
DRAM_OUTPUT_BASE = 0x0800
VMEM_ACTIVATION_BASE = 0x2000
VMEM_WEIGHT_BASE = 0x2400
VMEM_OUTPUT_BASE = 0x2800

# Mock data for matmul verification
ACTIVATION_DATA = torch.eye(32, 32, dtype=torch.float8_e4m3fn)
WEIGHT_DATA = (2 * torch.eye(32, 32, dtype=torch.float32)).to(torch.float8_e4m3fn)
MATMUL_RESULT = (ACTIVATION_DATA.to(torch.float32) @ WEIGHT_DATA.to(torch.float32)).to(
    torch.bfloat16
)


class MatmulProgram(Program):
    """
    Rewritten Matmul test using structured Args dataclasses.
    """

    instructions: list[Instruction] = [
        # DRAM_ACTIVATION_BASE = 0x0000 (Fits in addi)
        ADDI(rd=x(4), rs1=x(0), imm=0x000),
        # DRAM_WEIGHT_BASE = 0x0400 (Fits in addi: 1024)
        ADDI(rd=x(5), rs1=x(0), imm=0x400),
        # VMEM_ACTIVATION_BASE = 0x2000
        LUI(rd=x(1), imm=0x2),
        # VMEM_WEIGHT_BASE = 0x2400
        LUI(rd=x(2), imm=0x2),
        ADDI(rd=x(2), rs1=x(2), imm=0x400),
        # VMEM_OUTPUT_BASE = 0x2800
        # Note: 0x800 is -2048 in signed 12-bit.
        # To get 0x2800, we load 0x3000 (lui 3) then subtract 0x800.
        LUI(rd=x(3), imm=0x3),
        ADDI(rd=x(3), rs1=x(3), imm=-2048),
        # x6 = 1024 (size for vmem store)
        ADDI(rd=x(6), rs1=x(0), imm=1024),
        # set DMA base
        DMA_CONFIG_CH0(rs1=x(0)),
        DMA_WAIT_CH0(),
        # store activation into VMEM (vmem[x1] = activation)
        DMA_LOAD_CH0(rd=x(1), rs1=x(4), rs2=x(6)),
        # store weight into vmem (vmem[x2] = weight)
        DMA_LOAD_CH1(rd=x(2), rs1=x(5), rs2=x(6)),
        DMA_WAIT_CH0(),
        DMA_WAIT_CH1(),
        DELAY(imm=16),
        # load weights/activations from vmem
        VLOAD(vd=m(0), rs1=x(1), imm=0),  # mrf[v0] = activations
        VLOAD(vd=m(1), rs1=x(2), imm=0),  # mrf[v1] = weights
        DELAY(16),
        # push to weight buffer, matmul, and pop from accumulation buffer
        VMATPUSH_WEIGHT_MXU0(vd=w(0), vs1=m(1)),
        # VPU local transfer (1KB) is ~16 cycles at 64B/cycle
        DELAY(16),
        VMATMUL_MXU0(vd=acc(0), vs1=m(0), vs2=w(0)),
        # MXU matmul latency is 32 cycles by default; add small slack.
        DELAY(imm=32),
        VMATPOP_BF16_ACC_MXU0(vd=m(2), vs2=acc(0)),
        # store to vmem
        VSTORE(vd=m(2), rs1=x(3), imm=0),
        VSTORE(vd=m(3), rs1=x(3), imm=32),
        # Two vstores are ~2x16 cycles; add slack before DMA reads VMEM.
        DELAY(imm=16),
        # store to dram
        # DRAM_OUTPUT_BASE = 0x0800 (2048)
        # To get 0x800: LUI 1 (0x1000) + ADDI -2048 = 0x0800
        LUI(rd=x(10), imm=1),
        ADDI(rd=x(10), rs1=x(10), imm=-2048),
        ADDI(rd=x(11), rs1=x(10), imm=1024),
        # IMPORTANT: DMA ops read XRF at *execute* time. Since DMA now has non-trivial
        # latency (based on XRF[rs2] length), don't mutate x3 between the two stores.
        DMA_STORE_CH0(rd=x(10), rs1=x(3), rs2=x(6)),
        ADDI(rd=x(12), rs1=x(3), imm=1024),  # we cannot mutate x3 during the execution of the first dma store. as such, use x12
        DMA_STORE_CH1(rd=x(11), rs1=x(12), rs2=x(6)),
        DMA_WAIT_CH0(),
        DMA_WAIT_CH1(),
    ]

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
        (DRAM_WEIGHT_BASE, WEIGHT_DATA),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        torch.cat((MATMUL_RESULT[:, :16], MATMUL_RESULT[:, 16:]), dim=0),
    )
