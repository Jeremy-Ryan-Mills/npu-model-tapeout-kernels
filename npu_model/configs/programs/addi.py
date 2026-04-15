import torch
from npu_model.isa import *
from ...software import (
    acc, m, x, w,
    Program,
)

from ..isa_definition import ADDI, BLT, DELAY, VMATMUL_MXU1


class AddiProgram(Program):
    """
    A simple addi program with a branch and a matmul.
    """

    instructions: list[Instruction] = [
        ADDI(rd=x(1), rs1=x(1), imm=0),
        ADDI(rd=x(2), rs1=x(2), imm=8),
        ADDI(rd=x(1), rs1=x(1), imm=1),
        BLT(rs1=x(1), rs2=x(2), imm=-1),
        DELAY(imm=3),
        VMATMUL_MXU1(vd=acc(1), vs1=m(1), vs2=w(1)),
        ADDI(rd=x(4), rs1=x(4), imm=1),
        ADDI(rd=x(5), rs1=x(5), imm=1),
        DELAY(imm=0)
    ]

    memory_regions: list[tuple[int, torch.Tensor]] = []
