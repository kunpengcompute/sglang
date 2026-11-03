#!/usr/bin/env python3
"""Enable vision execution in a Qwen Kunpeng unified-checkpoint capability plan.

Unified w8a8 checkpoints may pin their component capability policy through
``checkpoint_loader_spec`` in ``config.json``: ``{"plan_path": ...,
"plan_sha256": ...}``. The plan JSON carries a ``capability_policy`` whose
SHA256 is verified at startup. Enabling vision for such a checkpoint requires
rewriting ``capability_policy`` to the approved vision-enabled policy and
refreshing the pinned SHA256.

Checkpoints without ``checkpoint_loader_spec`` need no plan: the runtime
policy comes from ``SGLANG_KUNPENG_QWEN_VISION=1`` instead.

Usage:
    python enable_vision_plan.py --model-dir /path/to/checkpoint [--mtp]
    python enable_vision_plan.py --plan plan.json --config config.json [--mtp]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from sglang.srt.model_executor.model_capabilities import (
        ATTACHED_MTP_COMPONENT_POLICY,
        STAGE_COMPONENT_POLICY,
        VISION_ATTACHED_MTP_COMPONENT_POLICY,
        VISION_STAGE_COMPONENT_POLICY,
    )
except Exception:  # pragma: no cover - standalone execution without sglang
    STAGE_COMPONENT_POLICY = {
        "language": {"execution_enabled": True, "pack_policy": "once"},
        "vision": {"execution_enabled": False, "pack_policy": "deferred"},
        "mtp": {"execution_enabled": False, "pack_policy": "deferred"},
    }
    ATTACHED_MTP_COMPONENT_POLICY = {
        "language": {"execution_enabled": True, "pack_policy": "once"},
        "vision": {"execution_enabled": False, "pack_policy": "deferred"},
        "mtp": {"execution_enabled": True, "pack_policy": "once"},
    }
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        help="Checkpoint directory containing config.json (auto-derives the plan path)",
    )
    parser.add_argument("--plan", help="Path to the capability plan JSON")
    parser.add_argument("--config", help="Path to the config.json carrying the spec")
    parser.add_argument(
        "--mtp",
        action="store_true",
        help="Also enable attached MTP in the plan (requires matching EAGLE startup)",
    )
    args = parser.parse_args()

    if args.model_dir is None and (args.plan is None or args.config is None):
        parser.error("provide --model-dir, or both --plan and --config")

    if args.model_dir is not None:
        config_path = Path(args.model_dir) / "config.json"
        if not config_path.is_file():
            print(f"error: {config_path} not found", file=sys.stderr)
            return 1
        config = json.loads(config_path.read_text())
        spec = config.get("checkpoint_loader_spec")
        if spec is None:
            print(
                "config.json has no checkpoint_loader_spec: no plan to regenerate. "
                "Set SGLANG_KUNPENG_QWEN_VISION=1 at server launch instead."
            )
            return 0
        plan_path = Path(spec["plan_path"])
    else:
        config_path = Path(args.config)
        config = json.loads(config_path.read_text())
        plan_path = Path(args.plan)

    if not plan_path.is_file():
        print(f"error: plan file {plan_path} not found", file=sys.stderr)
        return 1

    plan = json.loads(plan_path.read_text())
    current = plan.get("capability_policy")
    target = (
        VISION_ATTACHED_MTP_COMPONENT_POLICY if args.mtp else VISION_STAGE_COMPONENT_POLICY
    )
    known = (
        STAGE_COMPONENT_POLICY,
        ATTACHED_MTP_COMPONENT_POLICY,
        VISION_STAGE_COMPONENT_POLICY,
        VISION_ATTACHED_MTP_COMPONENT_POLICY,
    )
    if current is not None and not any(
        json.dumps(current, sort_keys=True) == json.dumps(k, sort_keys=True)
        for k in known
    ):
        print(
            f"error: plan carries an unrecognized capability_policy: {current}",
            file=sys.stderr,
        )
        return 1

    plan["capability_policy"] = target
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    spec = config.setdefault("checkpoint_loader_spec", {})
    spec["plan_path"] = str(plan_path)
    spec["plan_sha256"] = digest
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    print(f"plan: {plan_path}")
    print(f"sha256: {digest}")
    print(f"capability_policy: {json.dumps(target, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
