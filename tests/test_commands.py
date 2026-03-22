from __future__ import annotations

from typer.testing import CliRunner

from target_lock.commands import app as app_module
from target_lock.commands.config import MoveCommandConfig, TrackingConfig, VisionConfig
from target_lock.controllers import PidAimController
from target_lock.runner import AlignmentThreshold, BullseyeSource
from target_lock.runner import move as move_module


runner = CliRunner()


def test_move_command_loads_config_and_dispatches(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_move(config: MoveCommandConfig):
        captured["config"] = config
        return {"last_info": {}, "last_metrics": None}

    monkeypatch.setattr(app_module, "run_move", fake_run_move)
    app_module.move_pid(
        config=app_module.DEFAULT_MOVE_CONFIG_PATH,
        max_steps=5,
        fire=False,
    )

    config = captured["config"]
    assert isinstance(config, MoveCommandConfig)
    assert config.session.max_steps == 5
    assert config.session.fire_when_aligned is False


def test_move_command_can_enable_manual_base_control(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_move(config: MoveCommandConfig):
        captured["config"] = config
        return {"last_info": {}, "last_metrics": None}

    monkeypatch.setattr(app_module, "run_move", fake_run_move)

    app_module.move_pid(
        config=app_module.DEFAULT_MOVE_CONFIG_PATH,
        manual_base_control=True,
    )

    config = captured["config"]
    assert isinstance(config, MoveCommandConfig)
    assert config.motion.manual_base_control is True


def test_run_move_builds_runner_with_move_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}
    detector = object()

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["run_called"] = True
            return {"last_info": {}, "last_metrics": None}

    monkeypatch.setattr(move_module, "Runner", FakeRunner)
    monkeypatch.setattr(move_module, "build_bullseye_detector", lambda **kwargs: detector)

    move_module.run_move(MoveCommandConfig())

    assert captured["run_called"] is True
    assert isinstance(captured["controller"], PidAimController)
    assert captured["max_steps"] is None
    assert captured["fire_when_aligned"] is True
    assert captured["bullseye_source"] == BullseyeSource.VISION
    assert captured["bullseye_detector"] is detector
    assert captured["vision_detect_every_n_frames"] == 1
    assert captured["vision_smoothing_alpha"] == 1.0
    assert callable(captured["action_mutator"])
    assert captured["threshold"] == AlignmentThreshold(
        azimuth_deg=0.18,
        elevation_deg=0.18,
        plane_x=0.01,
        plane_y=0.01,
    )


def test_build_bullseye_detector_uses_async_wrapper_when_enabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncDetector:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(move_module, "AsyncCvBullseyeVision", FakeAsyncDetector)
    monkeypatch.setattr(move_module, "resolve_autoaim_onnx_path", lambda *_: "model.onnx")

    detector = move_module.build_bullseye_detector(
        tracking=TrackingConfig(bullseye_source="vision"),
        vision=VisionConfig(async_inference=True, smoothing_alpha=0.4),
    )

    assert isinstance(detector, FakeAsyncDetector)
    assert captured["onnx_path"] == "model.onnx"
    assert captured["smoothing_alpha"] == 0.4


def test_build_bullseye_detector_uses_sync_detector_when_async_disabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSyncDetector:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(move_module, "CvBullseyeVision", FakeSyncDetector)
    monkeypatch.setattr(move_module, "resolve_autoaim_onnx_path", lambda *_: "model.onnx")

    detector = move_module.build_bullseye_detector(
        tracking=TrackingConfig(bullseye_source="vision"),
        vision=VisionConfig(async_inference=False),
    )

    assert isinstance(detector, FakeSyncDetector)
    assert captured["onnx_path"] == "model.onnx"
    assert "smoothing_alpha" not in captured


def test_build_bullseye_detector_keeps_oracle_unwrapped(monkeypatch) -> None:
    sentinel = object()

    monkeypatch.setattr(move_module, "OracleBullseyeVision", lambda: sentinel)

    detector = move_module.build_bullseye_detector(
        tracking=TrackingConfig(bullseye_source="oracle"),
        vision=VisionConfig(async_inference=True),
    )

    assert detector is sentinel


def test_run_move_disables_runner_smoothing_when_async_enabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return {"last_info": {}, "last_metrics": None}

    monkeypatch.setattr(move_module, "Runner", FakeRunner)
    monkeypatch.setattr(move_module, "build_bullseye_detector", lambda **kwargs: object())

    config = MoveCommandConfig()
    config.vision.smoothing_alpha = 0.25
    config.vision.async_inference = True
    move_module.run_move(config)

    assert captured["vision_smoothing_alpha"] == 1.0


def test_run_move_preserves_runner_smoothing_when_async_disabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return {"last_info": {}, "last_metrics": None}

    monkeypatch.setattr(move_module, "Runner", FakeRunner)
    monkeypatch.setattr(move_module, "build_bullseye_detector", lambda **kwargs: object())

    config = MoveCommandConfig()
    config.vision.smoothing_alpha = 0.25
    config.vision.async_inference = False
    move_module.run_move(config)

    assert captured["vision_smoothing_alpha"] == 0.25
