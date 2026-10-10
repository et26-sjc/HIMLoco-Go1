"""Export one audited Quiet policy as a merged 270D-to-12D JIT module."""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'rsl_rl'))
sys.path.insert(0, str(REPO / 'legged_gym'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="quiet_policy.pt")
    parser.add_argument("--adapter", choices=("enabled", "disabled"), default="enabled")
    args = parser.parse_args()
    from rsl_rl.utils.quiet_checkpoint import load_quiet_policy, file_sha256
    model, checkpoint = load_quiet_policy(args.checkpoint, device="cpu",
                                          adapter_enabled=args.adapter == "enabled")
    output = os.path.abspath(args.output)
    model.export_policy(output, adapter_enabled=args.adapter == "enabled")
    metadata = {"source_checkpoint": os.path.abspath(args.checkpoint),
                "source_sha256": file_sha256(args.checkpoint),
                "source_iteration": int(checkpoint["iter"]), "adapter": args.adapter,
                "input_dim": model.policy_spec["num_actor_obs"],
                "output_dim": model.policy_spec["num_actions"],
                "action_scale": 0.25, "action_semantics": "joint target offsets",
                "policy": checkpoint["quiet_metadata"]["policy"]}
    with open(output + ".json", "w") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    print("Exported policy to " + output)


if __name__ == "__main__":
    main()
