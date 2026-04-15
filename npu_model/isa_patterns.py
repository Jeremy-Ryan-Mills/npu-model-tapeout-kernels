from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, Self

from .isa import UJType
from .isa_types import *

# Util Functions
scalar_offset_fmt = re.compile(
    r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)\(x(\d+)\)"
)
exponent_offset_fmt = re.compile(
    r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)\(e(\d+)\)"
)
matrix_offset_fmt = re.compile(
    r"^(0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+)\(m(\d+)\)"
)


def _wrong_cnt_error(
    mnemonic: str, expected: int, actual: int, operand_fmt: str
) -> AsmError:
    return AsmError(
        f"'{mnemonic}' expects {expected - 1} operand{'s' if expected != 2 else ''} ({operand_fmt}), got {actual - 1}",
        token_index=0,
    )


def _is_label_ref(s: str) -> bool:
    """Return True if s looks like a symbolic label (not a numeric literal)."""
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", s))


def _handle_offset_reg(
    token: str,
    immediate_format: ImmediateT = Imm12,
    reg_format: RegisterT = ScalarReg,
    role: str = "rs1",
    tok_idx: int = 0,
) -> list[AsmError]:
    expected = f"<{immediate_format.fmt}>({reg_format.fmt}<{role}>)"

    if reg_format is ScalarReg:
        match = scalar_offset_fmt.match(token)
    elif reg_format is ExponentReg:
        match = exponent_offset_fmt.match(token)
    elif reg_format is MatrixReg:
        match = matrix_offset_fmt.match(token)
    else:
        raise ValueError(
            f"Attempted to provide a non-register format to _is_offset_reg: {reg_format.__name__}"
        )

    if not match:
        return [
            AsmError(
                f"Expected base+offset operand in the form {expected}, got '{token}'",
                token_index=tok_idx,
            )
        ]

    imm = int(match.group(1), 0)
    reg = int(match.group(2))

    err = immediate_format.lint(imm, role="12-bit immediate", tok_idx=tok_idx)
    err.extend(reg_format.lint(reg, role=role, tok_idx=tok_idx))
    return err


class InstructionPattern(ABC):
    """
    Represents the ISA arguments for a specific instruction.

    This abstract class serves as the interface for parsing external assembly
    and providing the constructor for the Intermediate Representation (IR).
    It cannot be instantiated directly.

    All instruction patterns must define:
    - mnemonic (str): The name of the instruction. This is inferred by magic when
      paired with an InstructionType (which usees __init_subclass__ to set it.)
    - params (list[str]): The human-readable param format (for linting and errors)
    """

    mnemonic: ClassVar[str] = NotImplemented
    params: ClassVar[list[str]] = []

    @classmethod
    @abstractmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
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
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        """
        Validate whether a set of tokens can be used to instantiate an Instruction. Should be used
        on the instruction class itself (i.e. `li.lint`) NOT on the InstructionPattern class.

        Args:
            tokens: List of assembly instruction tokens (e.g., `["add", "x1", "x2", "x3"]`).
            labels: A list of labels that are available in the assembly.

        Returns:
            A list of Assembly Errors. If the list is empty, there are no problems with the instruction.

        Note:
            We do not check whether the jump to the label is too large. We just assume this is fine.
            If it's too large, it should get caught at assemble time.
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
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rd = cls.REG_TYPE(tokens[1])
        match = re.match(r"(\d+)\(x(\d+)\)", tokens[2])
        if not match:
            raise ValueError(
                f"Malformed set of tokens provided for {cls.mnemonic}: {','.join(tokens)}. Expected: {cls.mnemonic} {' '.join(cls.params)}"
            )

        imm = int(match.group(1))
        rs1 = ScalarReg(int(match.group(2)))

        return cls(rd=rd, imm=imm, rs1=rs1)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        if cls.REG_TYPE == ScalarReg:
            exceptions.extend(ScalarReg.lint(tokens[1], role="rd", tok_idx=1))
        else:
            exceptions.extend(ExponentReg.lint(tokens[1], role="rd", tok_idx=1))

        exceptions.extend(_handle_offset_reg(tokens[2], role="rs1", tok_idx=2))

        return exceptions

    def __init__(self, rd: Reg, imm: int, rs1: ScalarReg):
        self.rd = rd
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
    params = [f"{ScalarReg.fmt}<rd>", f"{Imm12.fmt}({ScalarReg.fmt}<rs1>)"]


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
    params = [f"{ExponentReg.fmt}<rd>", f"{Imm12.fmt}({ScalarReg.fmt}<rs1>)"]


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
    params = [f"{ScalarReg.fmt}<rs2>", f"{Imm12.fmt}({ScalarReg.fmt}<rs1>)"]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> ScalarBaseOffsetStore:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rs2 = ScalarReg(tokens[1])
        match = re.match(r"(\d+)\(x(\d+)\)", tokens[2])
        if not match:
            raise ValueError(
                f"Malformed set of tokens provided for {cls.mnemonic}: {','.join(tokens)}. Expected: {cls.mnemonic} {' '.join(cls.params)}"
            )

        imm = int(match.group(1))
        rs1 = ScalarReg(int(match.group(2)))

        return cls(rs2=rs2, imm=imm, rs1=rs1)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(ScalarReg.lint(tokens[1], role="rs2", tok_idx=1))
        exceptions.extend(_handle_offset_reg(tokens[2], role="rs1", tok_idx=2))

        return exceptions

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
    params = [f"{MatrixReg.fmt}<vd>", f"{Imm12.fmt}({ScalarReg.fmt}<rs1>)"]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = MatrixReg(tokens[1])
        match = re.match(r"(\d+)\(x(\d+)\)", tokens[2])
        if not match:
            raise ValueError(
                f"Malformed set of tokens provided for {cls.mnemonic}: {','.join(tokens)}. Expected: {cls.mnemonic} {' '.join(cls.params)}"
            )

        imm = int(match.group(1))
        rs1 = ScalarReg(int(match.group(2)))

        return cls(vd=vd, imm=imm, rs1=rs1)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(MatrixReg.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(_handle_offset_reg(tokens[2], role="rs1", tok_idx=2))

        return exceptions

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
    params = [f"{ScalarReg.fmt}<rd>", Imm20.fmt]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rd = ScalarReg(tokens[1])
        imm = resolve(tokens[2])

        return cls(rd=rd, imm=imm)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic, len(cls.params) + 1, len(tokens), " ".join(cls.params)
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(ScalarReg.lint(tokens[1], role="rd", tok_idx=1))
        if issubclass(cls, UJType) and _is_label_ref(tokens[2]):
            if tokens[2] not in labels:
                exceptions.append(
                    AsmError(f"Undefined label '{tokens[2]}'", token_index=2)
                )
        else:
            exceptions.extend(Imm20.lint(tokens[2], role="20-bit immediate", tok_idx=2))

        return exceptions

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
    params = [f"{ExponentReg.fmt}<rd>", Imm12.fmt]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rd = ExponentReg(tokens[1])
        imm = int(tokens[2], 0)

        return cls(rd=rd, imm=imm)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic, len(cls.params) + 1, len(tokens), " ".join(cls.params)
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(ExponentReg.lint(tokens[1], role="rd", tok_idx=1))
        exceptions.extend(Imm12.lint(tokens[2], role="12-bit immediate", tok_idx=2))

        return exceptions

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
    params = [f"{ScalarReg.fmt}<rd>", f"{ScalarReg.fmt}<rs1>", Imm12.fmt]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rd = ScalarReg(tokens[1])
        rs1 = ScalarReg(tokens[2])
        imm = int(tokens[3], 0)

        return cls(rd=rd, rs1=rs1, imm=imm)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(ScalarReg.lint(tokens[1], role="rd", tok_idx=1))
        exceptions.extend(ScalarReg.lint(tokens[2], role="rs1", tok_idx=2))
        exceptions.extend(Imm12.lint(tokens[3], role="12-bit immediate", tok_idx=3))

        return exceptions

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

    UPPER_IMM: int = 0b0000000
    rd: ScalarReg
    rs1: ScalarReg
    imm: Imm12
    params = [f"{ScalarReg.fmt}<rd>", f"{ScalarReg.fmt}<rs1>", Shamt.fmt]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rd = ScalarReg(tokens[1])
        rs1 = ScalarReg(tokens[2])
        imm = int(tokens[3], 0)

        return cls(rd=rd, rs1=rs1, imm=imm)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(ScalarReg.lint(tokens[1], role="rd", tok_idx=1))
        exceptions.extend(ScalarReg.lint(tokens[2], role="rs1", tok_idx=2))
        exceptions.extend(Shamt.lint(tokens[3], role="shift amount", tok_idx=3))

        return exceptions

    def __init__(self, rd: ScalarReg, rs1: ScalarReg, imm: int):
        self.rd = rd
        self.rs1 = rs1
        # Done like this on purpose so it doesn't need custom isa types.
        self.imm = Imm12((self.UPPER_IMM << 5) | Shamt(imm))


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
    params = [
        f"{ScalarReg.fmt}<rs1>",
        f"{ScalarReg.fmt}<rs2>",
        f"{SBImm12.fmt} or label",
    ]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rs1 = ScalarReg(tokens[1])
        rs2 = ScalarReg(tokens[2])
        imm = resolve(tokens[3])

        return cls(rs1=rs1, rs2=rs2, imm=imm)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(ScalarReg.lint(tokens[1], role="rs1", tok_idx=1))
        exceptions.extend(ScalarReg.lint(tokens[2], role="rs2", tok_idx=2))
        if _is_label_ref(tokens[3]):
            if tokens[3] not in labels:
                exceptions.append(
                    AsmError(f"Undefined label '{tokens[3]}'", token_index=3)
                )
        else:
            exceptions.extend(SBImm12.lint(tokens[3], role="branch offset", tok_idx=3))

        return exceptions

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
    params = [f"{MatrixReg.fmt}<vd>", Imm16.fmt]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = MatrixReg(tokens[1])
        imm = int(tokens[2], 0)

        return cls(vd=vd, imm=imm)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic, len(cls.params) + 1, len(tokens), " ".join(cls.params)
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(MatrixReg.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(Imm16.lint(tokens[2], role="16-bit immediate", tok_idx=2))

        return exceptions

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
    params = [f"{ScalarReg.fmt}<rd>", f"{ScalarReg.fmt}<rs1>", f"{ScalarReg.fmt}<rs2>"]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rd = ScalarReg(tokens[1])
        rs1 = ScalarReg(tokens[2])
        rs2 = ScalarReg(tokens[3])

        return cls(rd=rd, rs1=rs1, rs2=rs2)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(ScalarReg.lint(tokens[1], role="rd", tok_idx=1))
        exceptions.extend(ScalarReg.lint(tokens[2], role="rs1", tok_idx=2))
        exceptions.extend(ScalarReg.lint(tokens[3], role="rs2", tok_idx=3))

        return exceptions


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
    params = [f"{MatrixReg.fmt}<vd>", f"{MatrixReg.fmt}<vs1>"]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = MatrixReg(tokens[1])
        vs1 = MatrixReg(tokens[2])

        return cls(vd=vd, vs1=vs1)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(MatrixReg.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(MatrixReg.lint(tokens[2], role="vs1", tok_idx=2))

        return exceptions


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
    params = [f"{MatrixReg.fmt}<vd>", f"{MatrixReg.fmt}<vs1>", f"{MatrixReg.fmt}<vs2>"]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = MatrixReg(tokens[1])
        vs1 = MatrixReg(tokens[2])
        vs2 = MatrixReg(tokens[3])

        return cls(vd=vd, vs1=vs1, vs2=vs2)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(MatrixReg.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(MatrixReg.lint(tokens[2], role="vs1", tok_idx=2))
        exceptions.extend(MatrixReg.lint(tokens[3], role="vs2", tok_idx=3))

        return exceptions


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
    params = [
        f"{MatrixReg.fmt}<vd>",
        f"{MatrixReg.fmt}<vs2>",
        f"{ExponentReg.fmt}<es1>",
    ]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = MatrixReg(tokens[1])
        vs2 = MatrixReg(tokens[2])
        es1 = ExponentReg(tokens[3])

        return cls(vd=vd, vs2=vs2, es1=es1)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(MatrixReg.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(MatrixReg.lint(tokens[2], role="vs2", tok_idx=2))
        exceptions.extend(ExponentReg.lint(tokens[3], role="es1", tok_idx=3))

        return exceptions


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

    vd: WeightBuffer
    vs1: MatrixReg
    params = [f"{WeightBuffer.fmt}<vd>", f"{MatrixReg.fmt}<vs1>"]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = WeightBuffer(tokens[1])
        vs1 = MatrixReg(tokens[2])

        return cls(vd=vd, vs1=vs1)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(WeightBuffer.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(MatrixReg.lint(tokens[2], role="vs1", tok_idx=2))

        return exceptions


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

    vd: Accumulator
    vs1: MatrixReg
    params = [f"{Accumulator.fmt}<vd>", f"{MatrixReg.fmt}<vs1>"]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = Accumulator(tokens[1])
        vs1 = MatrixReg(tokens[2])

        return cls(vd=vd, vs1=vs1)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(Accumulator.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(MatrixReg.lint(tokens[2], role="vs1", tok_idx=2))

        return exceptions


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
    vs2: Accumulator
    params = [
        f"{MatrixReg.fmt}<vd>",
        f"{ExponentReg.fmt}<es1>",
        f"{Accumulator.fmt}<vs2>",
    ]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = MatrixReg(tokens[1])
        vs2 = Accumulator(tokens[3])
        es1 = ExponentReg(tokens[2])

        return cls(vd=vd, es1=es1, vs2=vs2)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(MatrixReg.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(ExponentReg.lint(tokens[2], role="es1", tok_idx=2))
        exceptions.extend(Accumulator.lint(tokens[3], role="vs2", tok_idx=3))

        return exceptions


@dataclass()
class MXUAccumulatorPop(InstructionPattern):
    """
    Instruction pattern for MXU data pop instructions with two tensor register operands.

    Matches assembly patterns of the form `instr m(vd), a(vs2)`. This pattern is
    used for MXU data push operations with one destination and one source tensor register.

    Attributes:
        vd: The destination tensor register.
        vs2: The source tensor register.
    """

    vd: MatrixReg
    vs2: Accumulator
    params = [f"{MatrixReg.fmt}<vd>", f"{Accumulator.fmt}<vs2>"]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = MatrixReg(tokens[1])
        vs2 = Accumulator(tokens[2])

        return cls(vd=vd, vs2=vs2)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(MatrixReg.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(Accumulator.lint(tokens[2], role="vs2", tok_idx=2))
        return exceptions


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

    vd: Accumulator
    vs1: MatrixReg
    vs2: WeightBuffer
    params = [
        f"{Accumulator.fmt}<vd>",
        f"{MatrixReg.fmt}<vs1>",
        f"{WeightBuffer.fmt}<vs2>",
    ]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        vd = Accumulator(tokens[1])
        vs1 = MatrixReg(tokens[2])
        vs2 = WeightBuffer(tokens[3])

        return cls(vd=vd, vs1=vs1, vs2=vs2)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic,
                    len(cls.params) + 1,
                    len(tokens),
                    " ".join(cls.params),
                )
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(Accumulator.lint(tokens[1], role="vd", tok_idx=1))
        exceptions.extend(MatrixReg.lint(tokens[2], role="vs1", tok_idx=2))
        exceptions.extend(WeightBuffer.lint(tokens[3], role="vs2", tok_idx=3))

        return exceptions


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
    params = [f"{ScalarReg.fmt}<rs1>"]

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        rs1 = ScalarReg(tokens[1])

        return cls(rs1=rs1)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(cls.mnemonic, len(cls.params) + 1, len(tokens), " ".join(cls.params))
            ]

        exceptions: list[AsmError] = []
        exceptions.extend(ScalarReg.lint(tokens[1], role="rs1", tok_idx=1))
        return exceptions


@dataclass()
class Nullary(InstructionPattern):
    params = []
    """
    Instruction pattern for nullary instructions with no operands.

    Matches assembly patterns of the form `instr`. This pattern is
    used for instructions with no arguments.
    """

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        return cls()

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [
                _wrong_cnt_error(
                    cls.mnemonic, len(cls.params) + 1, len(tokens), " ".join(cls.params)
                )
            ]
        return []


@dataclass(init=False)
class UnaryImm(InstructionPattern):
    params = [Imm12.fmt]
    """
    Instruction pattern for instructions with one immediate operand.

    Matches assembly patterns of the form `instr imm`. This pattern is
    used for `delay`.

    Attributes:
        imm: A 12-bit immediate
    """

    imm: Imm12

    @classmethod
    def from_asm(
        cls, tokens: list[str], resolve: Callable[[str], int] = lambda x: int(x, 0)
    ) -> Self:
        if tokens[0] != cls.mnemonic:
            raise ValueError(
                f"Attempted to construct {cls.mnemonic} with tokens for {tokens[0]}."
            )

        imm = int(tokens[1], 0)

        return cls(imm=imm)

    @classmethod
    def lint(cls, tokens: list[str], labels: list[str]) -> list[AsmError]:
        if len(tokens) != len(cls.params) + 1:
            return [_wrong_cnt_error(cls.mnemonic, len(cls.params) + 1, len(tokens), " ".join(cls.params))]

        exceptions: list[AsmError] = []
        exceptions.extend(Imm12.lint(tokens[1], role="12-bit immediate", tok_idx=1))
        return exceptions

    def __init__(self, imm: int):
        self.imm = Imm12(imm)
