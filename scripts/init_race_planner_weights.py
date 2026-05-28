#!/usr/bin/env python3
"""Write initial race planner weights (feedforward scales + optional MLP head)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from track_bonus.planner import default_race_ff_params, init_race_mlp_weights, save_planner_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs" / "race_planner_weights.npz",
    )
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = dict(default_race_ff_params())
    payload["mlp_residual_scale"] = 0.18
    if args.include_mlp:
        payload.update(init_race_mlp_weights(hidden=int(args.mlp_hidden), seed=int(args.seed)))
    save_planner_weights(args.output.resolve(), payload)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
