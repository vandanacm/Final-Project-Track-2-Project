import json

import numpy as np
import pytest

from competition.track_scene import build_track_model, resolve_go2_asset_model_dir
from go2_pg_env.track import StandardOvalTrack
from run_track_bonus import _validate_checkpoint, _validate_planner_track, official_track
from track_bonus.controller_interface import (
    LOWLEVEL_ACTION_SIZE,
    LOWLEVEL_STATE_OBS_SIZE,
    TRACK_OBS_FEATURE_NAMES,
    build_track_controller_observation,
    validate_high_level_command,
)
from track_bonus.planner import StarterPlannerConfig, StarterTrackPlanner


def test_validate_high_level_command_keeps_values_and_validates_shape() -> None:
    command = validate_high_level_command(np.asarray([2.0, -1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(command, np.asarray([2.0, -1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError):
        validate_high_level_command(np.asarray([0.1, 0.2], dtype=np.float32))
    with pytest.raises(ValueError):
        validate_high_level_command(np.asarray([0.1, np.nan, 0.2], dtype=np.float32))


def test_track_controller_observation_is_compact_track_state() -> None:
    track = StandardOvalTrack()
    xy, heading, _ = track.centerline_pose(0.0)
    qpos = np.zeros(19, dtype=np.float32)
    qpos[:2] = xy
    qpos[2] = 0.31
    qpos[3:7] = np.asarray([np.cos(0.5 * heading), 0.0, 0.0, np.sin(0.5 * heading)], dtype=np.float32)
    obs = build_track_controller_observation(qpos=qpos, track=track)
    assert TRACK_OBS_FEATURE_NAMES == (
        "lap_fraction",
        "lateral_error_norm",
        "boundary_margin_norm",
        "heading_error_rad",
        "curvature_norm",
    )
    assert obs.as_array().shape == (5,)
    assert abs(obs.lateral_error_norm) < 1e-6
    assert obs.boundary_margin_norm == pytest.approx(1.0)
    assert abs(obs.heading_error_rad) < 1e-6
    assert 0.0 <= obs.lap_fraction < 1.0


def test_planner_entry_point_uses_track_observation() -> None:
    planner = StarterTrackPlanner(StarterPlannerConfig(stand_seconds=0.0))
    track = StandardOvalTrack()
    xy, heading, _ = track.centerline_pose(5.0)
    qpos = np.zeros(19, dtype=np.float32)
    qpos[:2] = xy
    qpos[3:7] = np.asarray([np.cos(0.5 * heading), 0.0, 0.0, np.sin(0.5 * heading)], dtype=np.float32)
    obs = build_track_controller_observation(qpos=qpos, track=track)
    command = planner.command(obs, t=1.0)
    assert command.shape == (3,)
    assert np.all(np.isfinite(command))


def test_race_ff_planner_loads_learned_weights(tmp_path) -> None:
    weights = tmp_path / "planner_weights.npz"
    np.savez(
        weights,
        speed_mps=np.asarray(0.8),
        min_speed_mps=np.asarray(0.2),
        max_lateral_speed_mps=np.asarray(0.15),
        max_yaw_rate_radps=np.asarray(0.7),
        k_heading=np.asarray(0.9),
        k_lateral=np.asarray(0.12),
        turn_speed_drop=np.asarray(0.3),
        margin_power=np.asarray(0.6),
        curvature_feedforward=np.asarray(1.0),
        stand_seconds=np.asarray(0.0),
    )
    cfg = tmp_path / "planner_config.json"
    cfg.write_text(
        json.dumps(
            {
                "planner_type": "race_ff",
                "weights_path": "planner_weights.npz",
                "track_length_m": 200.0,
                "turn_radius_m": 18.25,
                "half_width_m": 2.0,
            }
        ),
        encoding="utf-8",
    )
    planner = StarterTrackPlanner.load(cfg)
    track = StandardOvalTrack()
    xy, heading, _ = track.centerline_pose(0.0)
    qpos = np.zeros(19, dtype=np.float32)
    qpos[:2] = xy
    qpos[3:7] = np.asarray([np.cos(0.5 * heading), 0.0, 0.0, np.sin(0.5 * heading)], dtype=np.float32)
    obs = build_track_controller_observation(qpos=qpos, track=track)
    command = planner.command(obs, t=1.0)
    assert command[0] > 0.1
    assert np.all(np.isfinite(command))


def test_track_scene_compiles_single_dog_when_assets_are_available() -> None:
    try:
        resolve_go2_asset_model_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    model = build_track_model(num_dogs=1, colors=["#2563EB"])
    assert model.nq == 19
    assert model.nu == 12


def test_student_track_scene_rejects_multi_dog_rendering() -> None:
    with pytest.raises(ValueError, match="one Go2"):
        build_track_model(num_dogs=2)


def test_checkpoint_validation_requires_hw1_actor_contract(tmp_path) -> None:
    config_path = tmp_path / "ppo_network_config.json"
    config_path.write_text(
        (
            '{"action_size": 12, '
            '"network_factory_kwargs": {"policy_obs_key": "state"}, '
            '"observation_size": {"state": {"shape": [48]}}}'
        ),
        encoding="utf-8",
    )
    _validate_checkpoint(tmp_path)

    config_path.write_text(
        (
            '{"action_size": 12, '
            '"network_factory_kwargs": {"policy_obs_key": "privileged_state"}, '
            '"observation_size": {"state": {"shape": [48]}}}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="policy_obs_key"):
        _validate_checkpoint(tmp_path)

    config_path.write_text(
        (
            f'{{"action_size": {LOWLEVEL_ACTION_SIZE}, '
            '"network_factory_kwargs": {"policy_obs_key": "state"}, '
            f'"observation_size": {{"state": {{"shape": [{LOWLEVEL_STATE_OBS_SIZE + 1}]}}}}}}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="state observation shape"):
        _validate_checkpoint(tmp_path)

    config_path.write_text(
        (
            f'{{"action_size": {LOWLEVEL_ACTION_SIZE + 1}, '
            '"network_factory_kwargs": {"policy_obs_key": "state"}, '
            f'"observation_size": {{"state": {{"shape": [{LOWLEVEL_STATE_OBS_SIZE}]}}}}}}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="action_size"):
        _validate_checkpoint(tmp_path)


def test_planner_track_validation_locks_official_scene() -> None:
    track = official_track()

    class Config:
        def __init__(self, half_width_m: float) -> None:
            self.half_width_m = half_width_m

        def to_dict(self) -> dict[str, float]:
            return {
                "track_length_m": 200.0,
                "turn_radius_m": 18.25,
                "half_width_m": self.half_width_m,
            }

    class Planner:
        def __init__(self, half_width_m: float) -> None:
            self.config = Config(half_width_m)
            self.track = StandardOvalTrack(half_width_m=half_width_m)

    good = Planner(2.0)
    _validate_planner_track(good, track)

    bad = Planner(3.0)
    with pytest.raises(ValueError, match="official track"):
        _validate_planner_track(bad, track)
