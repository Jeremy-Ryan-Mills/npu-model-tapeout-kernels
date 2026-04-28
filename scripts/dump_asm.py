#!/usr/bin/env python3
"""Dump a smolvla Program's instruction list as assembly text to stdout.

Usage:
    uv run scripts/dump_asm.py <ProgramClassName>
"""
import sys

from npu_model.configs.isa_definition import *  # noqa
from npu_model.isa import ScalarArgs, VectorArgs, MatrixArgs, DmaArgs

UNARY_VPU = {
    "vexp", "vsqrt", "vrecip", "vsquare", "vneg", "vabs", "vlog", "vsign",
    "vredmax.row", "vredsum.row",
}

def _is_unary(mnemonic: str) -> bool:
    return any(mnemonic.startswith(u) for u in UNARY_VPU)

def _is_matrix_mnemonic(mnemonic: str) -> bool:
    return any(k in mnemonic for k in ("vmatpush", "vmatmul", "vmatpop"))

def _fmt_instr(mnemonic: str, args, labels: dict[int, str], idx: int) -> str:
    ch = mnemonic.replace("<N>", str(args.channel)) if "<N>" in mnemonic and isinstance(args, DmaArgs) else mnemonic

    if isinstance(args, ScalarArgs):
        if ch == "delay":
            return f"delay {args.imm}"
        if ch == "lui":
            return f"lui x{args.rd}, 0x{args.imm:x}"
        if ch in ("addi", "andi", "ori", "xori", "slti", "sltiu", "slli", "srli", "srai"):
            return f"{ch} x{args.rd}, x{args.rs1}, {args.imm}"
        if ch in ("add", "sub", "and", "or", "xor", "slt", "sltu", "sll", "srl", "sra", "mul"):
            return f"{ch} x{args.rd}, x{args.rs1}, x{args.rs2}"
        if ch in ("blt", "bge", "beq", "bne", "bltu", "bgeu"):
            target = idx + args.imm
            dest = labels.get(target, str(args.imm * 4))
            return f"{ch} x{args.rs1}, x{args.rs2}, {dest}"
        if ch in ("ecall", "ebreak", "nop"):
            return ch
        return f"{ch} x{args.rd}, x{args.rs1}, {args.imm}"

    # MatrixArgs: used by fused_attention and other kernels
    if isinstance(args, MatrixArgs):
        if "vmatpush.weight" in ch:
            return f"{ch} w{args.vd}, m{args.vs1}"
        if "vmatpush.acc" in ch:
            return f"{ch} acc{args.vd}, m{args.vs1}"
        if "vmatmul" in ch:
            return f"{ch} acc{args.vd}, m{args.vs1}, w{args.vs2}"
        if "vmatpop.bf16" in ch:
            return f"{ch} m{args.vd}, acc{args.vs1}"
        if "vmatpop.fp8" in ch:
            # note: should take e0 scaling factor - known perf model bug
            return f"{ch} m{args.vd}, acc{args.vs1}"

    if isinstance(args, VectorArgs):
        if "vload" in ch:
            return f"{ch} m{args.vd}, {args.imm12}(x{args.rs1})"
        if "vstore" in ch:
            return f"{ch} m{args.vd}, {args.imm12}(x{args.rs1})"
        if ch.startswith("vli"):
            return f"{ch} m{args.vd}, {args.imm}"
        if ch.startswith("vmov"):
            return f"{ch} m{args.vd}, m{args.vs1}"
        # VectorArgs used for matrix ops (smolvla_matmul style)
        if _is_matrix_mnemonic(ch):
            if "vmatpush.weight" in ch:
                return f"{ch} w{args.vd}, m{args.vs1}"
            if "vmatpush.acc" in ch:
                return f"{ch} acc{args.vd}, m{args.vs1}"
            if "vmatmul" in ch:
                return f"{ch} acc{args.vd}, m{args.vs1}, w{args.vs2}"
            if "vmatpop.bf16" in ch:
                return f"{ch} m{args.vd}, acc{args.vs1}"
            if "vmatpop.fp8" in ch:
                # note: should take e0 scaling factor - known perf model bug
                return f"{ch} m{args.vd}, acc{args.vs1}"
        if _is_unary(ch):
            return f"{ch} m{args.vd}, m{args.vs1}"
        return f"{ch} m{args.vd}, m{args.vs1}, m{args.vs2}"

    if isinstance(args, DmaArgs):
        if "dma.config" in ch or "dma.wait" in ch:
            return ch
        return f"{ch} x{args.rd}, x{args.rs1}, x{args.rs2}"

    return f"# UNKNOWN: {ch} {args}"


def dump_instructions(instructions: list) -> str:
    # Collect branch targets to know where to place labels
    labels: dict[int, str] = {}
    counter = [0]
    for idx, instr in enumerate(instructions):
        if instr.mnemonic in ("blt", "bge", "beq", "bne", "bltu", "bgeu"):
            target = idx + instr.args.imm
            if target not in labels:
                counter[0] += 1
                labels[target] = f"loop_{counter[0]}"

    lines = []
    for idx, instr in enumerate(instructions):
        if idx in labels:
            lines.append(f"{labels[idx]}:")
        lines.append(_fmt_instr(instr.mnemonic, instr.args, labels, idx))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/dump_asm.py <ProgramClassName>")
        sys.exit(1)

    name = sys.argv[1]
    # Import lazily to avoid circular import
    mod_map = {
        "SmolVLAAttentionProgram":       "npu_model.configs.programs.smolvla_attention",
        "SmolVLAElementwiseAddProgram":  "npu_model.configs.programs.smolvla_elementwise_add",
        "SmolVLAElementwiseDivProgram":  "npu_model.configs.programs.smolvla_elementwise_div",
        "SmolVLAElementwiseMulProgram":  "npu_model.configs.programs.smolvla_elementwise_mul",
        "SmolVLAElementwiseSubProgram":  "npu_model.configs.programs.smolvla_elementwise_sub",
        "SmolVLAFusedAttentionProgram":  "npu_model.configs.programs.smolvla_fused_attention",
        "SmolVLAFusedMatmulBiasProgram": "npu_model.configs.programs.smolvla_fused_matmul_bias",
        "SmolVLAFusedNormScaleProgram":  "npu_model.configs.programs.smolvla_fused_norm_scale",
        "SmolVLAFusedSiluGateProgram":   "npu_model.configs.programs.smolvla_fused_silu_gate",
        "SmolVLAGeluTanhProgram":        "npu_model.configs.programs.smolvla_gelu_tanh",
        "SmolVLAMatmulProgram":          "npu_model.configs.programs.smolvla_matmul",
        "SmolVLAMatmulKChainProgram":    "npu_model.configs.programs.smolvla_matmul_k_chain",
        "SmolVLAReductionSumProgram":    "npu_model.configs.programs.smolvla_reduction_sum",
        "SmolVLARequantProgram":         "npu_model.configs.programs.smolvla_requant",
        "SmolVLARmsNormProgram":         "npu_model.configs.programs.smolvla_rms_norm",
        "SmolVLARopeFrequencyProgram":   "npu_model.configs.programs.smolvla_rope_frequency",
        "SmolVLASiluProgram":            "npu_model.configs.programs.smolvla_silu",
        "SmolVLASoftmaxProgram":         "npu_model.configs.programs.smolvla_softmax",
    }
    if name not in mod_map:
        print(f"Unknown program: {name}")
        sys.exit(1)
    import importlib
    mod = importlib.import_module(mod_map[name])
    cls = getattr(mod, name)
    print(dump_instructions(cls.instructions))
