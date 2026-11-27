"""Checkpoint-driven CPU Qwen mapping, including padded INT8 BA execution."""

import re


_BA = re.compile(r"(?:model\.language_model|model)\.layers\.\d+\.linear_attn\.in_proj_[ab](?:\.weight)?$")


def requires_padded_ba_scheme(prefix):
    return bool(_BA.fullmatch(prefix))


def mapping_for_description(rules, description):
    """Add only known FLOAT->W8A8 variants; unknown keys still fail loading."""
    rules = {key: dict(value) for key, value in rules.items()}
    for name, mode in description.items():
        if mode != "W8A8_DYNAMIC" or not name.endswith(".weight"):
            continue
        key = re.sub(r"\.(layers|blocks)\.\d+\.", r".\1.{index}.", name)
        rule = rules.get(key)
        if rule is None or rule["source_dtype"] != "BF16":
            continue
        for suffix in ("weight_scale", "weight_offset"):
            if description.get(name.removesuffix("weight") + suffix) != mode:
                raise ValueError(f"Missing quant description for {name}: {suffix}")
        # Existing allocation/loader ABI supports I8 rows and F32 row scales.
        rule["source_dtype"] = rule["dtype"] = "I8"
        if rule["component"] == "vision":
            bias_rule = rules.get(key.removesuffix("weight") + "bias")
            if bias_rule is not None:
                # ModelSlim exports the quantized vision linear's bias as F32
                # even though its description entry remains labelled FLOAT.
                bias_rule["source_dtype"] = bias_rule["dtype"] = "F32"
        for suffix in ("weight_scale", "weight_offset"):
            scale_rule = dict(rule)
            scale_rule.update(
                target=rule["target"].removesuffix("weight") + suffix,
                source_dtype="F32", dtype="F32",
                source_shape=[rule["source_shape"][0], 1],
            )
            rules[key.removesuffix("weight") + suffix] = scale_rule
    return rules
