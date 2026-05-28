#!/usr/bin/env python3
"""Tiny black-box search scaffold for the track bonus high-level planner.

This is intentionally simple and intentionally not a solved planner. It searches
over a small JSON controller by repeatedly running `run_track_bonus.py --no-render`.
Use it to debug the evaluation loop. For a leaderboard submission, replace the
planner internals with a learned policy that maps the official 5D observation
to `[vx, vy, yaw_rate]`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from track_bonus.planner import StarterPlannerConfig


ROOT = Path(__file__).resolve().parent


SEARCH_KEYS = [
    "speed_mps",
    "max_lateral_speed_mps",
    "max_yaw_rate_radps",
    "k_heading",
    "k_lateral",
    "heading_slowdown",
]


BOUNDS = {
    "speed_mps": (0.20, 1.05),
    "max_lateral_speed_mps": (0.03, 0.28),
    "max_yaw_rate_radps": (0.12, 1.05),
    "k_heading": (0.20, 1.60),
    "k_lateral": (0.02, 0.30),
    "heading_slowdown": (0.0, 0.80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-planner-config", type=Path, default=ROOT / "configs" / "starter_planner.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "course_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--eval-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clip_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    clipped = dict(candidate)
    for key, (low, high) in BOUNDS.items():
        clipped[key] = float(np.clip(float(clipped[key]), low, high))
    clipped["min_speed_mps"] = min(float(clipped.get("min_speed_mps", 0.12)), float(clipped["speed_mps"]))
    return clipped


def _sample_candidate(center: dict[str, Any], scale: float, rng: np.random.Generator) -> dict[str, Any]:
    candidate = dict(center)
    for key in SEARCH_KEYS:
        low, high = BOUNDS[key]
        sigma = scale * (high - low)
        candidate[key] = float(center[key] + rng.normal(0.0, sigma))
    return _clip_candidate(candidate)


def _run_eval(
    *,
    checkpoint_dir: Path,
    planner_path: Path,
    config: Path,
    output_dir: Path,
    eval_seconds: float,
    force_cpu: bool,
) -> float:
    cmd = [
        sys.executable,
        "run_track_bonus.py",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--planner-config",
        str(planner_path),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--duration-seconds",
        str(eval_seconds),
        "--no-render",
    ]
    if force_cpu:
        cmd.append("--force-cpu")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        return float(payload["scores"]["composite_score"])
    except Exception:
        return -1.0


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base = StarterPlannerConfig.load(args.base_planner_config).to_dict()
    center = _clip_candidate(base)
    best = dict(center)
    best_score = -1.0
    history = []

    for iteration in range(int(args.iterations)):
        scale = max(0.04, 0.18 * (0.72**iteration))
        candidates = [dict(best if best_score >= 0.0 else center)]
        while len(candidates) < int(args.population):
            candidates.append(_sample_candidate(best if best_score >= 0.0 else center, scale, rng))

        for candidate_idx, candidate in enumerate(candidates):
            candidate_dir = output_dir / "candidates" / f"iter_{iteration:02d}_cand_{candidate_idx:02d}"
            planner_path = candidate_dir / "planner_config.json"
            _write_json(planner_path, candidate)
            score = _run_eval(
                checkpoint_dir=args.checkpoint_dir,
                planner_path=planner_path,
                config=args.config,
                output_dir=candidate_dir / "eval",
                eval_seconds=float(args.eval_seconds),
                force_cpu=bool(args.force_cpu),
            )
            record = {"iteration": iteration, "candidate": candidate_idx, "score": score, "planner": candidate}
            history.append(record)
            if score > best_score:
                best_score = score
                best = dict(candidate)
                _write_json(output_dir / "best_planner_config.json", best)
                _write_json(output_dir / "best_score.json", {"score": best_score, "iteration": iteration, "candidate": candidate_idx})
            print(f"iter={iteration} cand={candidate_idx} score={score:.3f} best={best_score:.3f}", flush=True)

        _write_json(output_dir / "search_summary.json", {"best_score": best_score, "best_planner": best, "history": history})

    print(json.dumps({"best_score": best_score, "best_planner_config": str(output_dir / "best_planner_config.json")}, indent=2))


if __name__ == "__main__":
    main()
