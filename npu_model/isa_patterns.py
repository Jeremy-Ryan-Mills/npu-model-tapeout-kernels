from __future__ import annotations
from typing import Self, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
from .isa_types import *

# Util Functions
scalar_offset_fmt = re.compile(r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)\(x(\d+)\)")
exponent_offset_fmt = re.compile(r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)\(e(\d+)\)")
matrix_offset_fmt = re.compile(r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)\(m(\d+)\)")

def _is_offset_reg(token: str, immediate_format: ImmediateT = Imm12, reg_format: RegisterT = ScalarReg) -> bool:
    if reg_format is ScalarReg:
        match = scalar_offset_fmt.match(token)
    elif reg_format is ExponentReg:
        match = exponent_offset_fmt.match(token)
    elif reg_format is MatrixReg:
        match = matrix_offset_fmt.match(token)
    else:
        raise ValueError(f"Attempted to provide a non-register format to _is_offset_reg: {reg_format.__name__}")

    if not match:
        return False
    
    imm = int(match.group(1), 0)
    reg = int(match.group(2))

    return immediate_format.accepts(imm) and reg_format.accepts(reg)
    
class InstructionPattern(ABC):
    """
    Represents the ISA arguments for a specific instruction.

    This abstract class serves as the interface for parsing external assembly 
    and providing the constructor for the Intermediate Representation (IR). 
    It cannot be instantiated directly.
    """
    mnemonic: str = NotImplemented

    @classmethod
    @abstractmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        """
        Parse assembly instruction tokens and instantiate an instruction if the given set 
        of tokens represent a valid instance of the instruction. Should be used on the 
        instruction class itself (i.e. `li.from_asm`) NOT on the InstructionPattern class.
        
        Args:
            tokens: List of assembly instruction tokens (e.g., `["add", "x1", "x2", "x3"]`).
            resolve: Optionally, a label-resolution function if labels are supported.
        
        Returns:
            An instance of an Instruction parsed from the tokens.
        
        Raises:
            ValueError: When an invalid set of tokens is provided for a specific instruction.
            NotImplementedError: When called on the InstructionPattern class rather than a valid subclass.
        """
        raise NotImplementedError()
    
    @classmethod
    @abstractmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        """
        Validate whether a set of tokens can be used to instantiate an Instruction. Should be used 
        on the instruction class itself (i.e. `li.accepts`) NOT on the InstructionPattern class.
        
        Args:
            tokens: List of assembly instruction tokens (e.g., `["add", "x1", "x2", "x3"]`).
            resolve: Optionally, a label-resolution function if labels are supported.
        
        Returns:
            True if `from_asm` should succeed, False otherwise
        
        Raises:
            ValueError: When a malformed register is provided (i.e. )
            NotImplementedError: When called on the InstructionPattern class rather than a valid subclass.
        """
        raise NotImplementedError()

@dataclass(init=False)
class _OffsetLoad[Reg: (ScalarReg, ExponentReg)](InstructionPattern):
    """
    Instruction pattern for instructions with a destination and base-offset operand.

    Matches assembly patterns of the form `instr x(rd), imm(x(rs1))` or `instr e(rd), imm(x(rs1))`.
    This pattern is used for the following instructions: `lb`, `lh`, `lw`, `jalr`, and `seld`

    Attributes:
        rd: The destination scalar register.
        imm: A 12-bit immediate value representing the offset.
        rs1: The base scalar register.
    """
    REG_TYPE: type[Reg]
    rd: Reg
    imm: Imm12
    rs1: ScalarReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        rd = cls.REG_TYPE(tokens[1])
        match = re.match(r"(\d+)\(x(\d+)\)", tokens[2])
        if not match:
            raise ValueError(f"Malformed set of tokens provided for {cls.mnemonic}: {','.join(tokens)}")
        
        imm = int(match.group(1))
        rs1 = ScalarReg(int(match.group(2)))

        return cls(
           rd=rd,
           imm=imm,
           rs1=rs1
        )

    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and ScalarReg.accepts(tokens[1]) and _is_offset_reg(tokens[2])


    def __init__(self, rd: Reg, imm: int, rs1: ScalarReg):
        self.rd  = rd
        self.imm = Imm12(imm)
        self.rs1 = rs1

class ScalarOffsetLoad(_OffsetLoad[ScalarReg]):
    """
    Instruction pattern for instructions with a scalar destination register and base-offset operand.

    Matches assembly patterns of the form `instr x(rd), imm(x(rs1))`.
    This pattern is used for the following instructions: `lb`, `lh`, `lw`, and `jalr`.

    Attributes:
        rd: The destination scalar register.
        imm: A 12-bit immediate value representing the offset.
        rs1: The base scalar register.
    """
    REG_TYPE = ScalarReg
class ExponentOffsetLoad(_OffsetLoad[ExponentReg]):
    """
    Instruction pattern for instructions with a exponent destination register and base-offset operand.

    Matches assembly patterns of the form `instr e(rd), imm(x(rs1))`.
    This pattern is used for the following instructions: `seld`

    Attributes:
        rd: The destination scalar register.
        imm: A 12-bit immediate value representing the offset.
        rs1: The base scalar register.
    """
    REG_TYPE = ExponentReg

@dataclass(init=False)
class ScalarBaseOffsetStore(InstructionPattern):
    """
    Instruction pattern for instructions with a source and base-offset operand.

    Matches assembly patterns of the form `instr x(rs2), imm(x(rs1))`. This pattern is 
    used for the following scalar instructions: `sb`, `sh`, and `sw`.

    Attributes:
        rs2: The source scalar register.
        imm: A 12-bit immediate value representing the offset.
        rs1: The base scalar register.
    """
    rs2: ScalarReg
    imm: Imm12
    rs1: ScalarReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> ScalarBaseOffsetStore:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        rs2 = ScalarReg(tokens[1])
        match = re.match(r"(\d+)\(x(\d+)\)", tokens[2])
        if not match:
            raise ValueError(f"Malformed set of tokens provided for {cls.mnemonic}: {','.join(tokens)}")
        
        imm = int(match.group(1))
        rs1 = ScalarReg(int(match.group(2)))

        return cls(
           rs2=rs2,
           imm=imm,
           rs1=rs1
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and ScalarReg.accepts(tokens[1]) and _is_offset_reg(tokens[2])

    def __init__(self, rs2: ScalarReg, imm: int, rs1: ScalarReg):
        self.rs2 = rs2
        self.imm = Imm12(imm)
        self.rs1 = rs1

@dataclass(init=False)
class TensorBaseOffset(InstructionPattern):
    """
    Instruction pattern for instructions with a tensor destination/source and base-offset operand.

    Matches assembly patterns of the form `instr m(vd), imm(x(rs1))`. This pattern is 
    used for the following tensor instructions: `vload` and `vstore`.

    Attributes:
        vd: The destination/source tensor register.
        imm: A 12-bit immediate value representing the offset.
        rs1: The base scalar register.
    """
    vd: MatrixReg
    imm: Imm12
    rs1: ScalarReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = MatrixReg(tokens[1])
        match = re.match(r"(\d+)\(x(\d+)\)", tokens[2])
        if not match:
            raise ValueError(f"Malformed set of tokens provided for {cls.mnemonic}: {','.join(tokens)}")
        
        imm = int(match.group(1))
        rs1 = ScalarReg(int(match.group(2)))

        return cls(
           vd=vd,
           imm=imm,
           rs1=rs1
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and MatrixReg.accepts(tokens[1]) and _is_offset_reg(tokens[2])

    def __init__(self, vd: MatrixReg, imm: int, rs1: ScalarReg):
        self.vd = vd
        self.imm = Imm12(imm)
        self.rs1 = rs1

@dataclass(init=False)
class ScalarImm(InstructionPattern):
    """
    Instruction pattern for instructions with a destination register and immediate operand.

    Matches assembly patterns of the form `instr x(rd), imm`. This pattern 
    is used for the following instructions: `lui`, `auipc`, `jal`.

    Attributes:
        rd: The destination scalar register.
        imm: A 20-bit immediate value.
    """
    rd: ScalarReg
    imm: Imm20

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        rd = ScalarReg(tokens[1])
        imm = resolve(tokens[2])

        return cls(
            rd=rd,
            imm=imm
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and ScalarReg.accepts(tokens[1]) and Imm20.accepts(resolve(tokens[2]))

    def __init__(self, rd: ScalarReg, imm: int):
        self.rd = rd
        self.imm = Imm20(imm)

@dataclass(init=False)
class ExponentImm(InstructionPattern):
    """
    Instruction pattern for instructions with a exponent destination register and immediate operand.

    Matches assembly patterns of the form `instr e(rd), imm`. This pattern is used for the following 
    instructions: `seli`.

    Attributes:
        rd: The destination exponent register.
        imm: A 12-bit immediate value.
    """
    rd: ExponentReg
    imm: Imm12

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        rd = ExponentReg(tokens[1])
        imm = int(tokens[2], 0)

        return cls(
            rd=rd,
            imm=imm
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and ExponentReg.accepts(tokens[1]) and Imm12.accepts(tokens[2])

    def __init__(self, rd: ExponentReg, imm: int):
        self.rd = rd
        self.imm = Imm12(imm)

@dataclass(init=False)
class ScalarComputeImm(InstructionPattern):
    """
    Instruction pattern for instructions with a destination register, source register, and immediate operand.

    Matches assembly patterns of the form `instr x(rd), x(rs1), imm`. This pattern is 
    used for scalar arithmetic and logical instructions with immediate operands.

    Attributes:
        rd: The destination scalar register.
        rs1: The source scalar register.
        imm: A 12-bit immediate value.
    """
    rd: ScalarReg
    rs1: ScalarReg
    imm: Imm12

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        rd = ScalarReg(tokens[1])
        rs1 = ScalarReg(tokens[2])
        imm = int(tokens[3], 0)

        return cls(
            rd=rd,
            rs1=rs1,
            imm=imm
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 4 and tokens[0] == cls.mnemonic and ScalarReg.accepts(tokens[1]) and ScalarReg.accepts(tokens[2]) and Imm12.accepts(tokens[3])

    def __init__(self, rd: ScalarReg, rs1: ScalarReg, imm: int):
        self.rd = rd
        self.rs1 = rs1
        self.imm = Imm12(imm)

@dataclass(init=False)
class ScalarComputeShamt(InstructionPattern):
    """
    Instruction pattern for the shift operator. Restricts immediate size to shamt.

    Matches assembly patterns of the form `instr x(rd), x(rs1), shamt`.

    Attributes:
        rd: The destination scalar register.
        rs1: The source scalar register.
        shamt: A 5-bit immediate value.
    """
    rd: ScalarReg
    rs1: ScalarReg
    imm: Imm12

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        rd = ScalarReg(tokens[1])
        rs1 = ScalarReg(tokens[2])
        imm = int(tokens[3], 0)

        return cls(
            rd=rd,
            rs1=rs1,
            imm=imm
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 4 and tokens[0] == cls.mnemonic and ScalarReg.accepts(tokens[1]) and ScalarReg.accepts(tokens[2]) and Shamt.accepts(tokens[3])

    def __init__(self, rd: ScalarReg, rs1: ScalarReg, imm: int):
        self.rd = rd
        self.rs1 = rs1
        # Done like this on purpose so it doesn't need custom isa types.
        self.imm = Imm12(Shamt(imm)) 

@dataclass(init=False)
class ScalarBranchImm(InstructionPattern):
    """
    Instruction pattern for branches.

    Matches assembly patterns of the form `instr x(rs1), x(rs2), imm`.

    Attributes:
        rs1: A source scalar register.
        rs2: A source scalar register.
        imm: A 12-bit immediate value.
    """
    rs1: ScalarReg
    rs2: ScalarReg
    imm: SBImm12

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        rs1 = ScalarReg(tokens[1])
        rs2 = ScalarReg(tokens[2])
        imm = int(tokens[3], 0)

        return cls(
            rs1=rs1,
            rs2=rs2,
            imm=imm
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 4 and tokens[0] == cls.mnemonic and ScalarReg.accepts(tokens[1]) and ScalarReg.accepts(tokens[2]) and Imm12.accepts(tokens[3])

    def __init__(self, rs1: ScalarReg, rs2: ScalarReg, imm: int):
        self.rs1 = rs1
        self.rs2 = rs2
        self.imm = SBImm12(imm)

@dataclass(init=False)
class DirectImm(InstructionPattern):
    """
    Instruction pattern for instructions with a register and immediate operand.

    Matches assembly patterns of the form `instr m(vd), imm`. This pattern is 
    used for VI instructions: `vli.all`, `vli.row`, `vli.col`, and `vli.one`.

    Attributes:
        reg: The register operand.
        imm: An immediate value.
    """
    vd: MatrixReg
    imm: Imm16

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = MatrixReg(tokens[1])
        imm = int(tokens[2], 0)

        return cls(
            vd=vd,
            imm=imm
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and MatrixReg.accepts(tokens[1]) and Imm16.accepts(tokens[2])

    def __init__(self, vd: MatrixReg, imm: int):
        self.vd = vd
        self.imm = Imm16(imm)

@dataclass()
class ScalarComputeReg(InstructionPattern):
    """
    Instruction pattern for instructions with three scalar register operands.

    Matches assembly patterns of the form `instr x(rd), x(rs1), x(rs2)`. This pattern is 
    used for scalar arithmetic and logical instructions with three register operands.

    Attributes:
        rd: The destination scalar register.
        rs1: The first source scalar register.
        rs2: The second source scalar register.
    """
    rd: ScalarReg
    rs1: ScalarReg
    rs2: ScalarReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        rd = ScalarReg(tokens[1])
        rs1 = ScalarReg(tokens[2])
        rs2 = ScalarReg(tokens[3])

        return cls(
            rd=rd,
            rs1=rs1,
            rs2=rs2
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 4 and tokens[0] == cls.mnemonic and ScalarReg.accepts(tokens[1]) and ScalarReg.accepts(tokens[2]) and ScalarReg.accepts(tokens[3])

@dataclass()
class TensorComputeUnary(InstructionPattern):
    """
    Instruction pattern for tensor instructions with two tensor register operands.

    Matches assembly patterns of the form `instr m(vd), m(vs1)`. This pattern is 
    used for unary tensor operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
    """
    vd: MatrixReg
    vs1: MatrixReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = MatrixReg(tokens[1])
        vs1 = MatrixReg(tokens[2])

        return cls(
            vd=vd,
            vs1=vs1
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and MatrixReg.accepts(tokens[1]) and MatrixReg.accepts(tokens[2])

@dataclass()
class TensorComputeBinary(InstructionPattern):
    """
    Instruction pattern for tensor instructions with three tensor register operands.

    Matches assembly patterns of the form `instr m(vd), m(vs1), m(vs2)`. This pattern is 
    used for binary tensor operations with one destination and two source tensor registers.

    Attributes:
        vd: The destination tensor register.
        vs1: The first source tensor register.
        vs2: The second source tensor register.
    """
    vd: MatrixReg
    vs1: MatrixReg
    vs2: MatrixReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = MatrixReg(tokens[1])
        vs1 = MatrixReg(tokens[2])
        vs2 = MatrixReg(tokens[3])

        return cls(
            vd=vd,
            vs1=vs1,
            vs2=vs2
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 4 and tokens[0] == cls.mnemonic and MatrixReg.accepts(tokens[1]) and MatrixReg.accepts(tokens[2]) and MatrixReg.accepts(tokens[3])

@dataclass()
class TensorComputeMixed(InstructionPattern):
    """
    Instruction pattern for tensor instructions with mixed tensor and exponent register operands.

    Matches assembly patterns of the form `instr m(vd), m(vs1), e(es1)`. This pattern is 
    used for tensor operations with one destination tensor register, one source tensor register, 
    and one source exponent register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
        es1: The source exponent register.
    """
    vd: MatrixReg
    vs2: MatrixReg
    es1: ExponentReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = MatrixReg(tokens[1])
        vs2 = MatrixReg(tokens[2])
        es1 = ExponentReg(tokens[3])

        return cls(
            vd=vd,
            vs2=vs2,
            es1=es1
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 4 and tokens[0] == cls.mnemonic and MatrixReg.accepts(tokens[1]) and MatrixReg.accepts(tokens[2]) and ExponentReg.accepts(tokens[3])

@dataclass()
class MXUWeightPush(InstructionPattern):
    """
    Instruction pattern for MXU data push instructions with two tensor register operands.

    Matches assembly patterns of the form `instr w(vd), m(vs1)`. This pattern is 
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
    """
    vd: WeightBufferIndex
    vs1: MatrixReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = WeightBufferIndex(tokens[1])
        vs1 = MatrixReg(tokens[2])

        return cls(
            vd=vd,
            vs1=vs1
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and WeightBufferIndex.accepts(tokens[1]) and MatrixReg.accepts(tokens[2])

@dataclass()
class MXUAccumulatorPush(InstructionPattern):
    """
    Instruction pattern for MXU data push instructions with two tensor register operands.

    Matches assembly patterns of the form `instr a(vd), m(vs1)`. This pattern is 
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
    """
    vd: AccumulatorIndex
    vs1: MatrixReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = AccumulatorIndex(tokens[1])
        vs1 = MatrixReg(tokens[2])

        return cls(
            vd=vd,
            vs1=vs1
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and AccumulatorIndex.accepts(tokens[1]) and MatrixReg.accepts(tokens[2])

@dataclass()
class MXUAccumulatorPopE1(InstructionPattern):
    """
    Instruction pattern for MXU data pop instructions with two tensor register operands.

    Matches assembly patterns of the form `instr m(vd), a(vs2), e(es1)`. This pattern is 
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
    """
    vd: MatrixReg
    es1: ExponentReg
    vs2: AccumulatorIndex

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = MatrixReg(tokens[1])
        vs2 = AccumulatorIndex(tokens[3])
        es1 = ExponentReg(tokens[2])

        return cls(
            vd=vd,
            es1=es1,
            vs2=vs2
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 4 and tokens[0] == cls.mnemonic and MatrixReg.accepts(tokens[1]) and AccumulatorIndex.accepts(tokens[2]) and ExponentReg.accepts(tokens[3])

@dataclass()
class MXUAccumulatorPop(InstructionPattern):
    """
    Instruction pattern for MXU data pop instructions with two tensor register operands.

    Matches assembly patterns of the form `instr m(vd), a(vs2)`. This pattern is 
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs1: The source tensor register.
    """
    vd: MatrixReg
    vs2: AccumulatorIndex

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = MatrixReg(tokens[1])
        vs2 = AccumulatorIndex(tokens[2])

        return cls(
            vd=vd,
            vs2=vs2
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and MatrixReg.accepts(tokens[1]) and AccumulatorIndex.accepts(tokens[2])

@dataclass()
class MXUMatMul(InstructionPattern):
    """
    Instruction pattern for MXU matrix multiplication.

    Matches assembly patterns of the form `instr a(vd), m(vs1), w(vs2)`. This pattern is 
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination accumulator buffer.
        vs1: The source matrix.
        vs2: The source weights
    """
    vd: AccumulatorIndex
    vs1: MatrixReg
    vs2: WeightBufferIndex

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        vd = AccumulatorIndex(tokens[1])
        vs1 = MatrixReg(tokens[2])
        vs2 = WeightBufferIndex(tokens[3])

        return cls(
            vd=vd,
            vs1=vs1,
            vs2=vs2
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 3 and tokens[0] == cls.mnemonic and AccumulatorIndex.accepts(tokens[1]) and MatrixReg.accepts(tokens[2]) and WeightBufferIndex.accepts(tokens[3])



@dataclass()
class DMARegUnary(InstructionPattern):
    """
    Instruction pattern for DMA instructions with one scalar register operand.

    Matches assembly patterns of the form `instr x(rs1)`. This pattern is 
    used for `dma.config.N`.

    Attributes:
        rs1: The source scalar register.
    """
    rs1: ScalarReg

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        rs1 = ScalarReg(tokens[1])

        return cls(
            rs1=rs1
        )
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 2 and tokens[0] == cls.mnemonic and ScalarReg.accepts(tokens[1])

@dataclass()
class Nullary(InstructionPattern):
    """
    Instruction pattern for nullary instructions with no operands.

    Matches assembly patterns of the form `instr`. This pattern is 
    used for instructions with no arguments.
    """

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")

        return cls()
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 1 and tokens[0] == cls.mnemonic

@dataclass(init=False)
class UnaryImm(InstructionPattern):
    """
    Instruction pattern for instructions with one immediate operand.

    Matches assembly patterns of the form `instr imm`. This pattern is 
    used for `delay`.

    Attributes:
        imm: A 12-bit immediate
    """
    imm: Imm12

    @classmethod
    def from_asm(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}.")
        
        imm = int(tokens[1],0)

        return cls(imm=imm)
    
    @classmethod
    def accepts(cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)) -> bool:
        return len(tokens) == 2 and tokens[0] == cls.mnemonic and Imm12.accepts(tokens[1])

    def __init__(self, imm: int):
        self.imm = Imm12(imm)
