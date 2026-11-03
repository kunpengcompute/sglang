"""Component capability state shared by model, loader and request adapters."""

STAGE_COMPONENT_POLICY = {
    "language": {"execution_enabled": True, "pack_policy": "once"},
    "vision": {"execution_enabled": False, "pack_policy": "deferred"},
    "mtp": {"execution_enabled": False, "pack_policy": "deferred"},
}
# Only an explicitly pinned child plan can select attached Qwen MTP execution.
ATTACHED_MTP_COMPONENT_POLICY = {
    "language": {"execution_enabled": True, "pack_policy": "once"},
    "vision": {"execution_enabled": False, "pack_policy": "deferred"},
    "mtp": {"execution_enabled": True, "pack_policy": "once"},
}
# Vision-execution stage: the vision tower dequantizes its checkpoint INT8
# weights to BF16 at load time (no AMX pack; the TP16 vision projections do
# not satisfy the 32-task tile plan and carry bias) and executes in BF16.
# Enabled by default; opt out with SGLANG_KUNPENG_QWEN_VISION=0. A
# regenerated pinned plan can also select it.
VISION_STAGE_COMPONENT_POLICY = {
    "language": {"execution_enabled": True, "pack_policy": "once"},
    "vision": {"execution_enabled": True, "pack_policy": "once"},
    "mtp": {"execution_enabled": False, "pack_policy": "deferred"},
}
VISION_ATTACHED_MTP_COMPONENT_POLICY = {
    "language": {"execution_enabled": True, "pack_policy": "once"},
    "vision": {"execution_enabled": True, "pack_policy": "once"},
    "mtp": {"execution_enabled": True, "pack_policy": "once"},
}
# Generic SGLang metadata registry; no model/native class is imported here.
MODEL_COMPONENT_POLICIES = {"Qwen3_5ForConditionalGeneration": STAGE_COMPONENT_POLICY}
MODEL_VISION_COMPONENT_POLICIES = {
    "Qwen3_5ForConditionalGeneration": VISION_STAGE_COMPONENT_POLICY
}


def _vision_execution_enabled():
    import os

    # Default-on for Qwen3.5 conditional-generation serving on 920F;
    # SGLANG_KUNPENG_QWEN_VISION=0 opts back into text-only serving.
    return os.environ.get("SGLANG_KUNPENG_QWEN_VISION", "1") != "0"


def default_kunpeng_qwen_policy():
    """Component policy used when no pinned checkpoint plan is present.

    Shared by the model constructor and the Kunpeng backend validator so the
    scheduler, tokenizer manager and model all agree on one default.
    """
    if _vision_execution_enabled():
        return {"language": True, "vision": True, "mtp": False}
    return {"language": True, "vision": False, "mtp": False}


class ModelCapabilityError(ValueError):
    pass


class ModelCapabilities:
    def __init__(self, components):
        if (
            type(components) is not dict
            or any(type(v) is not bool for v in components.values())
            or components
            not in (
                {"language": True, "vision": False, "mtp": False},
                {"language": True, "vision": False, "mtp": True},
                {"language": True, "vision": True, "mtp": False},
                {"language": True, "vision": True, "mtp": True},
            )
        ):
            raise ValueError("Unapproved initial component policy")
        self._policy = dict(components)
        self._states = {
            name: {
                "loaded": False,
                "validated": False,
                "execution_enabled": False,
                "state": "not_loaded",
            }
            for name in components
        }

    def loaded_validated(self, name):
        state = self._states[name]
        if state["loaded"] or state["validated"]:
            raise RuntimeError("Duplicate component completion")
        state.update(
            loaded=True,
            validated=True,
            execution_enabled=self._policy[name],
            state=(
                "loaded_executable" if self._policy[name] else "loaded_not_executable"
            ),
        )

    def require(self, name):
        state = self._states[name]
        if not state["execution_enabled"]:
            raise ModelCapabilityError(f'{name} execution disabled: {state["state"]}')

    def policy_enabled(self, name):
        return self._policy[name]

    def receipt(self):
        return {name: dict(state) for name, state in self._states.items()}

    def attach_disabled(self, root, name, shared_module_ids=()):
        if self._policy[name]:
            raise ValueError("Disabled hook applied to enabled component")
        for module in root.modules():
            if id(module) in shared_module_ids:
                continue
            module._runtime_component = name
            module._runtime_capabilities = self
            module.register_forward_pre_hook(
                lambda module, inputs, n=name: self.require(n)
            )

    def attach_enabled(self, root, name, shared_module_ids=()):
        if name != "mtp" or not self._policy[name]:
            raise ValueError("Only enabled attached MTP may use this hook")
        for module in root.modules():
            if id(module) in shared_module_ids:
                continue
            module._runtime_component = name
            module._runtime_capabilities = self
            module.register_forward_pre_hook(
                lambda module, inputs, n=name: self.require(n)
            )


def initial_policy_from_spec(spec):
    """One pinned component policy before modules, tokenizer or workers exist."""
    import hashlib, json
    from pathlib import Path

    if spec is None:
        return None
    if type(spec) is not dict:
        raise ValueError("Malformed checkpoint capability spec")
    raw = Path(spec["plan_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["plan_sha256"]:
        raise ValueError("Capability policy expected-data SHA mismatch")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate capability policy JSON key")
            result[key] = value
        return result

    plan = json.loads(raw, object_pairs_hook=pairs)
    policy = plan["capability_policy"]
    approved = (
        STAGE_COMPONENT_POLICY,
        ATTACHED_MTP_COMPONENT_POLICY,
        VISION_STAGE_COMPONENT_POLICY,
        VISION_ATTACHED_MTP_COMPONENT_POLICY,
    )
    if not any(
        json.dumps(policy, sort_keys=True) == json.dumps(expected, sort_keys=True)
        for expected in approved
    ):
        raise ValueError("Unapproved current execution capability policy")
    return {name: row["execution_enabled"] for name, row in policy.items()}


def validate_startup_capabilities(policy, server_args):
    if policy is None:
        return
    if getattr(server_args, "language_only", False) or getattr(
        server_args, "encoder_only", False
    ):
        raise ModelCapabilityError(
            "Unified checkpoint requires all component modules; partial architecture flags disabled"
        )
    if (
        not policy["mtp"]
        and getattr(server_args, "speculative_algorithm", None) is not None
    ):
        raise ModelCapabilityError(
            "mtp execution disabled: speculative runtime configuration unavailable"
        )
    if policy["mtp"]:
        algorithm = getattr(server_args, "speculative_algorithm", None)
        algorithm = getattr(algorithm, "value", algorithm)
        steps = getattr(server_args, "speculative_num_steps", None)
        tokens = getattr(server_args, "speculative_num_draft_tokens", None)
        if (
            algorithm != "EAGLE"
            or getattr(server_args, "speculative_eagle_topk", None) != 1
            or type(steps) is not int
            or steps not in (2, 3)
            or type(tokens) is not int
            or tokens != steps + 1
        ):
            raise ModelCapabilityError(
                "Attached Qwen MTP requires EAGLE topk1, draft steps2/3 and verify tokens=steps+1"
            )


def validate_request_capabilities(policy, request):
    if policy is None:
        return
    if not policy["vision"] and (
        any(
            getattr(request, field, None) is not None
            for field in ("image_data", "video_data", "audio_data")
        )
        or bool(getattr(request, "need_wait_for_mm_inputs", False))
    ):
        raise ModelCapabilityError(
            "vision execution disabled: image/video/multimodal request unsupported at current stage"
        )


def initial_policy_for_model_config(model_config):
    import os

    spec = getattr(model_config.hf_config, "checkpoint_loader_spec", None)
    if spec is not None:
        return initial_policy_from_spec(spec)
    # Normal text serving uses the architecture policy without a deployment
    # manifest. Reject disabled speculative startup before allocating workers.
    if (
        getattr(model_config, "quantization", None) != "modelslim"
        or os.environ.get("SGLANG_USE_CPU_920F") != "1"
        or os.environ.get("SGLANG_KUNPENG_QWEN") != "1"
    ):
        return None
    policies = [
        (
            MODEL_VISION_COMPONENT_POLICIES[name]
            if _vision_execution_enabled()
            else MODEL_COMPONENT_POLICIES[name]
        )
        for name in getattr(model_config.hf_config, "architectures", [])
        if name in MODEL_COMPONENT_POLICIES
    ]
    if not policies:
        return None
    if any(policy != policies[0] for policy in policies):
        raise ValueError("Conflicting architecture component capability policies")
    return {name: row["execution_enabled"] for name, row in policies[0].items()}
