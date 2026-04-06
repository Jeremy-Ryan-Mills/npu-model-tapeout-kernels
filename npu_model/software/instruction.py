from typing import Callable
from dataclasses import asdict, is_dataclass

from ..isa import IsaSpec, Args, ScalarArgs, VectorArgs

class Instruction:
    """
    An instruction in the program sequence.

    Attributes:
        id: Unique instruction ID
        mnemonic: The mnemonic of the instruction
        args: The arguments of the instruction
        delay: The delay of the instruction
    """

    def __init__(
        self,
        mnemonic: str,
        args: Args,
        delay: int = 0,
    ) -> None:
        self.mnemonic = mnemonic
        self.args = args
        self.delay = delay

    def __str__(self) -> str:
        args_dict = asdict(self.args) if is_dataclass(self.args) else self.args
        args_str = [f"{k}={v}" for k, v in args_dict.items()]
        return f"{self.mnemonic} {', '.join(args_str)}"

    def assemble(self) -> int:
        # find type from mnemonic table
        operation = IsaSpec.operations[self.mnemonic]

        # our first pass should correctly set these things in the IR,
        # but as a convenience feature for people writing in the IR
        # we fix args for shifts and breakpoint/ecall here
        if self.mnemonic == "ecall" and isinstance(self.args, ScalarArgs):
            self.args.imm = 0b000000000000
        elif self.mnemonic == "ebreak" and isinstance(self.args, ScalarArgs):
            self.args.imm = 0b000000000001
        elif (self.mnemonic == "srli" or self.mnemonic == "srai") and isinstance(self.args, ScalarArgs):
            self.args.imm = self.args.imm & 0b000000011111
        elif self.mnemonic == "srai" and isinstance(self.args, ScalarArgs):
            self.args.imm = (self.args.imm & 0b000000011111) | 0b0100000000000
        elif self.mnemonic == "fence" and isinstance(self.args, ScalarArgs):
            self.args.imm = 0b000000000000
        
        if isinstance(self.args, VectorArgs)and self.args.imm12 != 0:
            self.args.imm = self.args.imm12

        return operation.instruction_type.assemble(
            operation.opcode,
            operation.funct2,
            operation.funct3,
            operation.funct7,
            self.args
        )

class Uop:
    """
    A dynamic instruction instance that is executing in the simulation
    """

    _next_id: int = 0

    def __init__(self, insn: Instruction) -> None:
        self.id = Uop._next_id
        Uop._next_id += 1
        self.insn = insn

        self.dispatch_delay: int = 0
        """the number of dispatch stalling cycles left"""
        self.execute_delay: int = 0
        """the number of execute stalling cycles left"""

        self.execute_fn: Callable | None = None
