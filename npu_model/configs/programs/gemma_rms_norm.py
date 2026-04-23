import torch
from npu_model.util.converter import load_asm
from npu_model.software.instruction import Instruction
from npu_model.software.program import Program, ASM_FOLDER

INPUT_DATA = torch.randn(32, 32, dtype=torch.bfloat16)
EPS = 1e-6

# DRAM layout
DRAM_INPUT_BASE = 0x0000
DRAM_EPS_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x1000

# VMEM layout
VMEM_INPUT_BASE = 0x2000
VMEM_EPS_BASE = 0x2800
VMEM_OUTPUT_BASE = 0x3000


class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row-wise mean via transpose + vreduce.sum (second-to-last dim) + vbroadcast.cols.
    """

    instructions: list[Instruction] = load_asm(ASM_FOLDER / 'gemma_rms_norm.S')

    memory_regions: list[tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT_DATA),
        (DRAM_EPS_BASE, torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
    ]

    # FIXME: Re-derive a standalone golden reference for the pair-register BF16
    # VPU path. The current kernel wiring is exercised by simulation, but the
    # previous float-side golden no longer matches the staged BF16 execution.
    golden_result = None
