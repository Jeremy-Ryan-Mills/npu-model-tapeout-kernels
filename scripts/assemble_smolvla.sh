#!/usr/bin/env bash
# Generate .bin and .hex for all smolvla kernels from their Python program classes.
set -euo pipefail

BIN_DIR="npu_model/configs/programs/bin"
HEX_DIR="npu_model/configs/programs/hex"

mkdir -p "$BIN_DIR" "$HEX_DIR"

uv run python - <<'EOF'
import struct
from npu_model.configs.isa_definition import *  # noqa

KERNELS = [
    ("smolvla_attention",       "npu_model.configs.programs.smolvla_attention",       "SmolVLAAttentionProgram"),
    ("smolvla_elementwise_add", "npu_model.configs.programs.smolvla_elementwise_add", "SmolVLAElementwiseAddProgram"),
    ("smolvla_elementwise_div", "npu_model.configs.programs.smolvla_elementwise_div", "SmolVLAElementwiseDivProgram"),
    ("smolvla_elementwise_mul", "npu_model.configs.programs.smolvla_elementwise_mul", "SmolVLAElementwiseMulProgram"),
    ("smolvla_elementwise_sub", "npu_model.configs.programs.smolvla_elementwise_sub", "SmolVLAElementwiseSubProgram"),
    ("smolvla_fused_attention",  "npu_model.configs.programs.smolvla_fused_attention",  "SmolVLAFusedAttentionProgram"),
    ("smolvla_fused_matmul_bias","npu_model.configs.programs.smolvla_fused_matmul_bias","SmolVLAFusedMatmulBiasProgram"),
    ("smolvla_fused_norm_scale", "npu_model.configs.programs.smolvla_fused_norm_scale", "SmolVLAFusedNormScaleProgram"),
    ("smolvla_fused_silu_gate",  "npu_model.configs.programs.smolvla_fused_silu_gate",  "SmolVLAFusedSiluGateProgram"),
    ("smolvla_gelu_tanh",        "npu_model.configs.programs.smolvla_gelu_tanh",        "SmolVLAGeluTanhProgram"),
    ("smolvla_matmul",           "npu_model.configs.programs.smolvla_matmul",           "SmolVLAMatmulProgram"),
    ("smolvla_matmul_k_chain",   "npu_model.configs.programs.smolvla_matmul_k_chain",   "SmolVLAMatmulKChainProgram"),
    ("smolvla_reduction_sum",    "npu_model.configs.programs.smolvla_reduction_sum",    "SmolVLAReductionSumProgram"),
    ("smolvla_requant",          "npu_model.configs.programs.smolvla_requant",          "SmolVLARequantProgram"),
    ("smolvla_rms_norm",         "npu_model.configs.programs.smolvla_rms_norm",         "SmolVLARmsNormProgram"),
    ("smolvla_rope_frequency",   "npu_model.configs.programs.smolvla_rope_frequency",   "SmolVLARopeFrequencyProgram"),
    ("smolvla_silu",             "npu_model.configs.programs.smolvla_silu",             "SmolVLASiluProgram"),
    ("smolvla_softmax",          "npu_model.configs.programs.smolvla_softmax",          "SmolVLASoftmaxProgram"),
]

import importlib

for name, module_path, class_name in KERNELS:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    words = cls().assemble()

    bin_path = f"npu_model/configs/programs/bin/{name}.bin"
    hex_path = f"npu_model/configs/programs/hex/{name}.hex"

    with open(bin_path, "wb") as f:
        for w in words:
            f.write(struct.pack("<I", w & 0xFFFFFFFF))

    with open(hex_path, "w") as f:
        for w in words:
            f.write(f"{w & 0xFFFFFFFF:08x}\n")

    print(f"  {name}: {len(words)} words")
EOF
