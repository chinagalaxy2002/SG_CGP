"""Check that a native-binding checkpoint contains no method-specific parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")

    forbidden = sorted(
        key for key in state if "native_bind" in key.lower() or "native_binding" in key.lower()
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "state_dict_keys": len(state),
        "method_specific_parameter_keys": forbidden,
        "zero_parameter_native_binding": not forbidden,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if forbidden:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
