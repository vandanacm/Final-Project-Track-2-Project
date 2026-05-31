"""High-level planners for the 200 m track bonus.

The evaluator builds the official compact 5D track observation defined in
`track_bonus/controller_interface.py`. The high-level planner maps it to the
local joystick command consumed by the HW1 Go2 locomotion policy:

    5D track observation -> [vx, vy, yaw_rate]

Supported planner types (set in planner_config.json):

- ``starter_pd``: weak baseline PD controller (interface example).
- ``race_ff``: feedforward + Stanley-style PD with learned scale parameters (npz).
- ``race_mlp``: small MLP residual on top of ``race_ff`` with learned weights (npz).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"
    speed_mps: float = 0.45
    min_speed_mps: float = 0.12
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.25
    k_heading: float = 0.55
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0
    # race_ff / race_mlp
    weights_path: str = ""
    turn_speed_drop: float = 0.35
    margin_power: float = 0.65
    curvature_feedforward: float = 1.05
    mlp_hidden: int = 32
    mlp_residual_scale: float = 0.18
    track_length_m: float = 200.0
    turn_radius_m: float = 18.25
    half_width_m: float = 2.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _resolve_weights_path(config: StarterPlannerConfig, planner_config_path: Path | None) -> Path | None:
    if not str(config.weights_path).strip():
        return None
    weights = Path(config.weights_path)
    if weights.is_absolute():
        return weights
    if planner_config_path is not None:
        candidate = planner_config_path.parent / weights
        if candidate.exists():
            return candidate
    repo_candidate = Path(__file__).resolve().parent.parent / weights
    if repo_candidate.exists():
        return repo_candidate
    return weights


def default_race_ff_params() -> dict[str, float]:
    """Sub-90 s target: ~2.5 m/s straights, ~2.25 m/s in turns.

    Estimated lap time: straights 85.3m/2.5 ≈ 34s + turns 114.7m/2.25 ≈ 51s ≈ 85s.
    Assumes a low-level policy trained with stage_2 goal ranges (vx≤2.5, vy, yaw).
    """
    return {
        "speed_mps": 2.50,
        "min_speed_mps": 1.40,
        "max_lateral_speed_mps": 0.30,
        "max_yaw_rate_radps": 1.20,
        "k_heading": 1.20,
        "k_lateral": 0.22,
        "turn_speed_drop": 0.10,
        "margin_power": 0.40,
        "curvature_feedforward": 1.15,
        "stand_seconds": 0.0,
    }


def init_race_mlp_weights(hidden: int = 32, seed: int = 0) -> dict[str, np.ndarray]:
    """Xavier-initialized 5 -> hidden -> 3 MLP (residual head, near zero output)."""
    rng = np.random.default_rng(int(seed))
    w1 = rng.normal(0.0, np.sqrt(2.0 / 5.0), size=(5, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, 3)).astype(np.float32) * 0.05
    b2 = np.zeros(3, dtype=np.float32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def save_planner_weights(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in payload.items()})


class StarterTrackPlanner:
    """Track planner with optional learned parameters (npz) or MLP residual."""

    def __init__(
        self,
        config: StarterPlannerConfig,
        *,
        planner_config_path: Path | None = None,
    ) -> None:
        self.config = config
        self.track: StandardOvalTrack = official_track()
        self._planner_config_path = planner_config_path
        self._race_params = default_race_ff_params()
        self._mlp: dict[str, np.ndarray] | None = None
        weights_path = _resolve_weights_path(config, planner_config_path)
        if weights_path is not None and weights_path.exists():
            self._load_weights(weights_path)

        if config.planner_type not in {"starter_pd", "race_ff", "race_mlp"}:
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        return cls(StarterPlannerConfig.load(path), planner_config_path=path.resolve())

    def _load_weights(self, path: Path) -> None:
        data = dict(np.load(path, allow_pickle=False))
        for key in (
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
            "mlp_residual_scale",
        ):
            if key in data:
                self._race_params[key] = float(np.asarray(data[key]).reshape(()))
        if "w1" in data:
            self._mlp = {key: np.asarray(data[key], dtype=np.float32) for key in ("w1", "b1", "w2", "b2")}

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        stand = float(self._race_params.get("stand_seconds", self.config.stand_seconds))
        if t < stand:
            return np.zeros(3, dtype=np.float32)
        if self.config.planner_type == "starter_pd":
            return self._starter_pd_command(obs)
        return self._race_command(obs)

    def _starter_pd_command(self, obs: TrackControllerObservation) -> np.ndarray:
        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        lateral_bias = math.atan2(
            float(self.config.k_lateral) * lateral_error,
            max(float(self.config.speed_mps), 1e-3),
        )
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

        speed_scale = 1.0 - float(self.config.heading_slowdown) * min(abs(heading_error), math.pi) / math.pi
        vx = np.clip(
            float(self.config.speed_mps) * speed_scale,
            float(self.config.min_speed_mps),
            float(self.config.speed_mps),
        )
        vy = np.clip(
            -float(self.config.k_lateral) * lateral_error,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        yaw_rate = np.clip(
            curvature * vx + float(self.config.k_heading) * heading_error,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)

    def _race_command(self, obs: TrackControllerObservation) -> np.ndarray:
        p = self._race_params
        lateral_m = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        heading_error = float(obs.heading_error_rad)
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        margin = float(np.clip(obs.boundary_margin_norm, 0.0, 1.25))

        turn_factor = 1.0 - float(p["turn_speed_drop"]) * min(abs(float(obs.curvature_norm)), 1.0)
        margin_factor = float(margin**float(p["margin_power"]))
        heading_penalty = 1.0 - 0.35 * min(abs(heading_error), math.pi) / math.pi
        vx = float(p["speed_mps"]) * turn_factor * margin_factor * heading_penalty
        vx = float(np.clip(vx, float(p["min_speed_mps"]), float(p["speed_mps"])))

        vy = float(np.clip(
            -float(p["k_lateral"]) * lateral_m,
            -float(p["max_lateral_speed_mps"]),
            float(p["max_lateral_speed_mps"]),
        ))
        yaw_rate = float(
            float(p["curvature_feedforward"]) * curvature * vx + float(p["k_heading"]) * heading_error
        )
        yaw_rate = float(np.clip(yaw_rate, -float(p["max_yaw_rate_radps"]), float(p["max_yaw_rate_radps"])))
        command = np.asarray([vx, vy, yaw_rate], dtype=np.float32)

        if self.config.planner_type == "race_mlp" and self._mlp is not None:
            command = command + self._mlp_residual(obs) * float(
                p.get("mlp_residual_scale", self.config.mlp_residual_scale)
            )
            command[0] = float(np.clip(command[0], float(p["min_speed_mps"]), float(p["speed_mps"])))
            command[1] = float(np.clip(command[1], -float(p["max_lateral_speed_mps"]), float(p["max_lateral_speed_mps"])))
            command[2] = float(np.clip(command[2], -float(p["max_yaw_rate_radps"]), float(p["max_yaw_rate_radps"])))
        return command.astype(np.float32)

    def _mlp_residual(self, obs: TrackControllerObservation) -> np.ndarray:
        assert self._mlp is not None
        x = obs.as_array().astype(np.float32)
        h = np.tanh(x @ self._mlp["w1"] + self._mlp["b1"])
        return (h @ self._mlp["w2"] + self._mlp["b2"]).astype(np.float32)


# Backward-compatible alias used in docs.
TrackPlanner = StarterTrackPlanner
