"""Strip optimizer/trainer state from a Lightning checkpoint for GitHub release."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="identity-span-safe-dq-cgp")
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        raise KeyError(f"No state_dict in {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": payload["state_dict"],
            "epoch": payload.get("epoch"),
            "global_step": payload.get("global_step"),
            "source": args.name,
            "format": "weights-only",
        },
        args.output,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
