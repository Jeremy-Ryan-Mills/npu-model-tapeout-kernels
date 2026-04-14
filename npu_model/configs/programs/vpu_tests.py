from ...software import m, x, Program
from npu_model.isa import Instruction
import torch

from npu_model.configs.isa_definition import *

# Memory layout (DRAM is program-loaded; VMEM is scratchpad accessed by vload/vstore)
DRAM_INPUT_BASE = 0x0000
DRAM_OUTPUT_BASE = 0x0400
VMEM_INPUT_BASE = 0x2000
VMEM_OUTPUT_BASE = 0x2400

# One MRF register worth of bf16 data: (mrf_depth, mrf_width / bf16_bytes)
# With default configs this is typically 32x16 elements = 1024 bytes.
INPUT = torch.arange(32 * 16, dtype=torch.bfloat16).reshape(32, 16)


class VectorArithmeticProgram(Program):
    """
    Basic arithmetic correctness
    """

    instructions: list[Instruction] = [
        # Set up base addresses and transfer size (bytes)
        ADDI(rd=x(1), rs1=x(0), imm=VMEM_INPUT_BASE),
        ADDI(rd=x(2), rs1=x(0), imm=VMEM_OUTPUT_BASE),
        ADDI(rd=x(3), rs1=x(0), imm=DRAM_INPUT_BASE),
        ADDI(rd=x(4), rs1=x(0), imm=DRAM_OUTPUT_BASE),
        ADDI(rd=x(5), rs1=x(0), imm=1024),
        # DRAM -> VMEM
        DMA_CONFIG_CH0(rs1=x(0)),
        DMA_WAIT_CH0(),
        DMA_LOAD_CH0(rd=x(1), rs1=x(3), rs2=x(5)),
        DMA_WAIT_CH0(),
        DELAY(imm=16),
        # VMEM -> MRF, compute, MRF -> VMEM
        VLOAD(vd=m(0), rs1=x(1), imm=0),
        DELAY(imm=16),
        VADD_BF16(vd=m(1), vs1=m(0), vs2=m(0)),
        DELAY(imm=32),
        VSUB_BF16(vd=m(2), vs1=m(1), vs2=m(0)),
        DELAY(imm=32),
        VMUL_BF16(vd=m(3), vs1=m(2), vs2=m(0)),
        DELAY(imm=32),
        VSTORE(vd=m(3), rs1=x(2), imm=0),
        # Ensure the VPU has time to commit the VMEM write before DMA reads it.
        # There is currently no explicit VPU↔DMA memory ordering primitive in the model.
        DELAY(imm=16),
        # VMEM -> DRAM
        DMA_STORE_CH0(rd=x(4), rs1=x(2), rs2=x(5)),
        DMA_WAIT_CH0(),
    ]

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        (INPUT**2),
    )
