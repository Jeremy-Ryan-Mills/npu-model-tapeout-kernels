"""Compatibility shim for parameterized kernel code generators.

The parameterized kernels were written against the old ISA API which used
ScalarArgs/VectorArgs/MatrixArgs/DmaArgs dataclasses and a string-mnemonic
Instruction constructor. The cleanup branch replaced that with a class-based
ISA. The generator functions are still needed at module load time to produce
golden test data (memory_regions, golden_result), so we provide lightweight
stand-ins here. The instruction lists they produce are discarded — each
Program class loads its instructions from a pre-generated .S file instead.
"""

from dataclasses import dataclass, field


@dataclass
class ScalarArgs:
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0


@dataclass
class VectorArgs:
    vd: int = 0
    vs1: int = 0
    vs2: int = 0
    rs1: int = 0
    rs2: int = 0
    es1: int = 0
    base: int = 0
    offset: int = 0
    imm12: int = 0
    imm: int = 0


@dataclass
class MatrixArgs:
    vd: int = 0
    vs1: int = 0
    vs2: int = 0
    rd: int = 0
    rs1: int = 0
    rs2: int = 0


@dataclass
class DmaArgs:
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    size: int = 0
    channel: int = 0


class _MockInstruction:
    """Stub instruction used only inside parameterized generator functions.

    The output of those generators is discarded; Program classes load real
    instructions from .S files via load_asm().
    """
    __slots__ = ("mnemonic", "args")

    def __init__(self, mnemonic: str, args=None):
        self.mnemonic = mnemonic
        self.args = args

    def to_bytecode(self) -> int:
        return 0
