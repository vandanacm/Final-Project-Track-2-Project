#!/usr/bin/env python3
"""Evolution-style trainer for learned race planners (race_ff / race_mlp).

Maps official 5D track observations to [vx, vy, yaw_rate] by optimizing planner
weights against `run_track_bonus.py` rollout scores. Intended for Colab/GPU runs
with a HW1-compatible low-level checkpoint.

Example (feedforward scales only, fast):

    python train_highlevel_mlp.py \\
      --checkpoint-dir path/to/best_checkpoint \\
      --planner-type race_ff \\
      --output-dir artifacts/highlevel_race_ff \\
      --iterations 12 --population 16 --eval-seconds 90

Example (MLP residual on top of race_ff):

    python train_highlevel_mlp.py \\
      --checkpoint-dir path/to/best_checkpoint \\
      --planner-type race_mlp \\
      --output-dir artifacts/highlevel_race_mlp \\
      --iterations 20 --population 20 --eval-seconds 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from track_bonus.planner import (
    StarterPlannerConfig,
    default_race_ff_params,
    init_race_mlp_weights,
    save_planner_weights,
)


ROOT = Path(__file__).resolve().parent

RACE_FF_KEYS = [
    "speed_mps",
    "min_speed_mps",
    "max_lateral_speed_mps",
    "max_yaw_rate_radps",
    "k_heading",
    "k_lateral",
    "turn_speed_drop",
    "margin_power",
    "curvature_feedforward",
    "stand_seconds",
]

RACE_FF_BOUNDS: dict[str, tuple[float, float]] = {
    "speed_mps": (0.55, 3.20),
    "min_speed_mps": (0.20, 1.60),
    "max_lateral_speed_mps": (0.06, 0.30),
    "max_yaw_rate_radps": (0.35, 1.50),
    "k_heading": (0.35, 1.80),
    "k_lateral": (0.04, 0.32),
    "turn_speed_drop": (0.05, 0.60),
    "margin_power": (0.25, 1.10),
    "curvature_feedforward": (0.70, 1.60),
    "stand_seconds": (0.0, 0.80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "course_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--planner-type", choices=["race_ff", "race_mlp"], default="race_ff")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--eval-seconds", type=float, default=90.0)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--init-weights", type=Path, default=None, help="Optional npz to warm-start.")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clip_race_ff(params: dict[str, float]) -> dict[str, float]:
    out = dict(params)
    for key, (low, high) in RACE_FF_BOUNDS.items():
        out[key] = float(np.clip(float(out[key]), low, high))
    out["min_speed_mps"] = min(float(out["min_speed_mps"]), float(out["speed_mps"]) - 0.05)
    return out


def _vectorize_race_ff(params: dict[str, float]) -> np.ndarray:
    return np.asarray([float(params[k]) for k in RACE_FF_KEYS], dtype=np.float64)


def _devectorize_race_ff(vec: np.ndarray) -> dict[str, float]:
    return {key: float(vec[i]) for i, key in enumerate(RACE_FF_KEYS)}


def _load_init_vector(args: argparse.Namespace) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    params = _clip_race_ff(default_race_ff_params())
    mlp = init_race_mlp_weights(hidden=int(args.mlp_hidden), seed=int(args.seed))
    if args.init_weights is not None and args.init_weights.exists():
        data = dict(np.load(args.init_weights, allow_pickle=False))
        for key in RACE_FF_KEYS:
            if key in data:
                params[key] = float(np.asarray(data[key]).reshape(()))
        params = _clip_race_ff(params)
        if "w1" in data:
            mlp = {k: np.asarray(data[k], dtype=np.float32) for k in ("w1", "b1", "w2", "b2")}
    return params, mlp


def _pack_genome(params: dict[str, float], mlp: dict[str, np.ndarray] | None, planner_type: str) -> np.ndarray:
    vec = [_vectorize_race_ff(params)]
    if planner_type == "race_mlp" and mlp is not None:
        for key in ("w1", "b1", "w2", "b2"):
            vec.append(mlp[key].reshape(-1).astype(np.float64))
    return np.concatenate(vec)


def _unpack_genome(
    genome: np.ndarray,
    *,
    center_params: dict[str, float],
    center_mlp: dict[str, np.ndarray] | None,
    planner_type: str,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    ff_dim = len(RACE_FF_KEYS)
    params = _devectorize_race_ff(genome[:ff_dim])
    params = _clip_race_ff(params)
    mlp = None
    if planner_type == "race_mlp" and center_mlp is not None:
        mlp = {k: np.array(center_mlp[k], copy=True) for k in center_mlp}
        offset = ff_dim
        for key in ("w1", "b1", "w2", "b2"):
            size = center_mlp[key].size
            mlp[key] = genome[offset : offset + size].reshape(center_mlp[key].shape).astype(np.float32)
            offset += size
    return params, mlp


def _save_candidate(
    *,
    output_dir: Path,
    params: dict[str, float],
    mlp: dict[str, np.ndarray] | None,
    planner_type: str,
    mlp_hidden: int,
    mlp_residual_scale: float,
) -> Path:
    weights_path = output_dir / "planner_weights.npz"
    payload: dict[str, Any] = dict(params)
    payload["mlp_residual_scale"] = float(mlp_residual_scale)
    if mlp is not None:
        payload.update(mlp)
    save_planner_weights(weights_path, payload)

    planner_cfg = {
        "planner_type": planner_type,
        "weights_path": "planner_weights.npz",
        "mlp_hidden": int(mlp_hidden),
        "mlp_residual_scale": float(mlp_residual_scale),
        "track_length_m": 200.0,
        "turn_radius_m": 18.25,
        "half_width_m": 2.0,
    }
    planner_path = output_dir / "planner_config.json"
    _write_json(planner_path, planner_cfg)
    return planner_path


def _run_eval(
    *,
    checkpoint_dir: Path,
    planner_path: Path,
    config: Path,
    output_dir: Path,
    eval_seconds: float,
    force_cpu: bool,
) -> dict[str, Any]:
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
        "--entry-name",
        "train_candidate",
    ]
    if force_cpu:
        cmd.append("--force-cpu")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fitness(payload: dict[str, Any]) -> float:
    if not payload:
        return -1.0
    scores = payload.get("scores", {})
    metrics = payload.get("metrics", {})
    composite = float(scores.get("composite_score", -1.0))
    completion = float(metrics.get("lap_completion", 0.0))
    finish = metrics.get("finish_time")
    # Tournament priority: finish lap first, then minimize time, else maximize distance.
    if completion >= 1.0 and finish is not None:
        return 1000.0 + completion * 10.0 - float(finish) / 300.0
    return composite + 2.5 * completion


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    center_params, center_mlp = _load_init_vector(args)
    if args.planner_type == "race_ff":
        center_mlp = None

    center_genome = _pack_genome(center_params, center_mlp, args.planner_type)
    best_genome = center_genome.copy()
    best_score = -1.0
    history: list[dict[str, Any]] = []

    mlp_residual_scale = 0.18

    for iteration in range(int(args.iterations)):
        scale = max(0.02, 0.12 * (0.75**iteration))
        candidates = [best_genome.copy() if best_score >= 0 else center_genome.copy()]
        while len(candidates) < int(args.population):
            noise = rng.normal(0.0, scale, size=center_genome.shape)
            if args.planner_type == "race_mlp":
                # Keep MLP perturbations smaller than feedforward gains.
                ff_dim = len(RACE_FF_KEYS)
                noise[ff_dim:] *= 0.35
            candidates.append(center_genome + noise)

        for cand_idx, genome in enumerate(candidates):
            params, mlp = _unpack_genome(
                genome,
                center_params=center_params,
                center_mlp=center_mlp,
                planner_type=args.planner_type,
            )
            cand_dir = output_dir / "candidates" / f"iter_{iteration:02d}_cand_{cand_idx:02d}"
            planner_path = _save_candidate(
                output_dir=cand_dir,
                params=params,
                mlp=mlp,
                planner_type=args.planner_type,
                mlp_hidden=int(args.mlp_hidden),
                mlp_residual_scale=mlp_residual_scale,
            )
            payload = _run_eval(
                checkpoint_dir=args.checkpoint_dir.resolve(),
                planner_path=planner_path,
                config=args.config.resolve(),
                output_dir=cand_dir / "eval",
                eval_seconds=float(args.eval_seconds),
                force_cpu=bool(args.force_cpu),
            )
            score = _fitness(payload)
            metrics = payload.get("metrics", {})
            record = {
                "iteration": iteration,
                "candidate": cand_idx,
                "fitness": score,
                "composite_score": payload.get("scores", {}).get("composite_score"),
                "lap_completion": metrics.get("lap_completion"),
                "finish_time": metrics.get("finish_time"),
                "valid_distance_m": metrics.get("valid_distance_m"),
                "params": params,
            }
            history.append(record)
            if score > best_score:
                best_score = score
                best_genome = genome.copy()
                best_params, best_mlp = params, mlp
                best_dir = output_dir / "best"
                best_planner = _save_candidate(
                    output_dir=best_dir,
                    params=best_params,
                    mlp=best_mlp,
                    planner_type=args.planner_type,
                    mlp_hidden=int(args.mlp_hidden),
                    mlp_residual_scale=mlp_residual_scale,
                )
                _write_json(best_dir / "best_metrics.json", payload)
                shutil_copy = best_planner.parent / "planner_config.json"
                _write_json(output_dir / "best_planner_config.json", json.loads(shutil_copy.read_text()))
                save_planner_weights(output_dir / "planner_weights.npz", dict(np.load(best_dir / "planner_weights.npz")))

            print(
                f"iter={iteration} cand={cand_idx} fitness={score:.3f} "
                f"lap={metrics.get('lap_completion')} dist={metrics.get('valid_distance_m')} "
                f"best={best_score:.3f}",
                flush=True,
            )

    _write_json(output_dir / "search_summary.json", {"best_fitness": best_score, "history": history})
    print(
        json.dumps(
            {
                "best_fitness": best_score,
                "best_planner_config": str(output_dir / "best_planner_config.json"),
                "best_weights": str(output_dir / "planner_weights.npz"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
