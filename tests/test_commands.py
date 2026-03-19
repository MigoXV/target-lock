from __future__ import annotations

from typer.testing import CliRunner

from target_lock.cli.common import AlignmentThreshold, BullseyeSource
from target_lock.commands import app as app_module
from target_lock.controllers import PidAimController


runner = CliRunner()


def test_move_command_uses_pid_controller_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}
    detector = object()

    def fake_run_session(**kwargs):
        captured.update(kwargs)
        return {"last_info": {}, "last_metrics": None}

    monkeypatch.setattr(app_module, "run_session", fake_run_session)
    monkeypatch.setattr(app_module, "_build_bullseye_detector", lambda **kwargs: detector)

    result = runner.invoke(app_module.app, ["move", "--max-steps", "5", "--no-fire"])

    assert result.exit_code == 0, result.stdout
    assert isinstance(captured["controller"], PidAimController)
    assert captured["max_steps"] == 5
    assert captured["fire_when_aligned"] is False
    assert captured["bullseye_source"] == BullseyeSource.VISION
    assert captured["bullseye_detector"] is detector
    assert captured["threshold"] == AlignmentThreshold(
        azimuth_deg=0.18,
        elevation_deg=0.18,
        plane_x=0.01,
        plane_y=0.01,
    )
