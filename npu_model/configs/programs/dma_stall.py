from npu_model.isa import Instruction
import torch
from ...software import (
    acc, e, m, x, w,
    Program,
)

from ..isa_definition import *

class DMAStallProgram(Program):
    """
    A simple program demonstrating DMA loads, stalling logic, and matrix multiplication
    updated for the latest npu_model ISA.
    """

    instructions: list[Instruction] = [
        # --- 1. Setup Scalar Registers ---
        # Set x2 = 1024 (Base Address & Size 1024)
        ADDI(rd=x(1), rs1=x(0), imm=0),
        ADDI(rd=x(2), rs1=x(0), imm=1024),
        # --- 2. Configure DMA Channels ---
        # Configure Base Address x1 (0)
        DMA_CONFIG_CH0(rs1=x(0)),
        DMA_WAIT_CH0(),
        # --- 3. Load thing ---
        # Load 1024 bytes (x2) from DRAM to VMEM(x1) on Channel 0
        # Note: rs1 specifies VMEM offset, rs2 specifies length. Base address comes from dma.config
        DMA_LOAD_CH0(rd=x(0), rs1=x(1), rs2=x(2)),
        # Load 1024 bytes (x2) from DRAM to VMEM(x2) on Channel 1
        DMA_LOAD_CH1(rd=x(1), rs1=x(2), rs2=x(2)),
        # Wait to get these things in VMEM
        DMA_WAIT_CH0(),
        DMA_WAIT_CH1(),
        # Move VMEM data to actual computational registers
        # vload VMEM(x1=0) -> MRF 2
        VLOAD(vd=m(1), rs1=x(0), imm=0),
        # vload VMEM(x2=1024) -> Temporary MRF 1
        VLOAD(vd=m(0), rs1=x(1), imm=0),
        DELAY(imm=100),
        # Push Temporary MRF 1 -> MXU0 Weight Buffer 1
        VMATPUSH_WEIGHT_MXU0(vd=w(0), vs1=m(0)),
        # --- 4. Do unnecessary loads (Overlapped with Matmul) ---
        # We do not need to reconfigure the DMA channels if the base addresses are unchanged
        # Issue DMA loads again (DRAM -> VMEM)
        DMA_LOAD_CH0(rd=x(3), rs1=x(0), rs2=x(1)),
        DMA_LOAD_CH1(rd=x(4), rs1=x(1), rs2=x(1)),
        # --- 5. Do matmul ---
        VMATMUL_MXU0(vd=acc(0), vs1=m(1), vs2=w(0)),
        DELAY(imm=32), # TODO - verify delays
        # VMATMUL.MXU0(vd=0, vs1=0, vs2=0),
        # VMATMUL.MXU0(vd=0, vs1=0, vs2=0),
        # VMATMUL.MXU0(vd=0, vs1=0, vs2=0),
        VMATPOP_FP8_ACC_MXU0(vd=m(0), es1=e(0), vs2=acc(0)),
        # DELAY(imm=32),
        # Wait to finish unnecessary loads
        DMA_WAIT_CH0(),
        DMA_WAIT_CH1(),
        # End delay
        DELAY(imm=0)
    ]

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (0, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
        (1024, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
    ]
