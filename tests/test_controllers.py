from __future__ import annotations

import numpy as np

from target_lock.cli.common import BullseyeSource, _clear_aim_action, _resolve_tracking_info
from target_lock.controllers import ActionLayout, OpenLoopAimConfig, OpenLoopAimController, PidAimConfig, PidAimController
from target_lock.vision import BullseyeDetection


def _info() -> dict[str, object]:
    return {
        "bullseye_pixel": [480, 120],
        "camera_fovy_deg": 60.0,
        "camera_fovx_deg": 80.0,
        "width": 640,
        "height": 480,
    }


def test_open_loop_writes_configured_axes() -> None:
    layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = OpenLoopAimController(OpenLoopAimConfig(yaw_step_rad=0.08, pitch_step_rad=0.08, action_layout=layout))

    action, metrics = controller.update(_info(), (480, 640, 3))

    assert action.shape == (6,)
    assert action[4] > 0.0
    assert action[3] < 0.0
    assert metrics.azimuth_deg > 0.0


def test_pid_controller_resets_when_target_missing() -> None:
    layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = PidAimController(
        PidAimConfig(
            open_loop=OpenLoopAimConfig(yaw_step_rad=0.08, pitch_step_rad=0.08, action_layout=layout),
        )
    )

    action, _ = controller.update(_info(), (480, 640, 3), dt=0.01)
    assert action.shape == (6,)
    scan_action, scan_metrics = controller.update({}, (480, 640, 3), dt=0.01)
    assert scan_action[3] > 0.0
    assert scan_action[4] == 0.0
    assert scan_metrics.as_dict()["scan_yaw_command"] > 0.0
    assert controller.yaw_pid.initialized is False


def test_clear_aim_action_preserves_non_aim_axes() -> None:
    layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    action = np.array([0.2, -0.3, 0.4, 0.9, -0.8, 1.0], dtype=np.float32)

    cleared = _clear_aim_action(action.copy(), layout)

    assert np.allclose(cleared[:3], np.array([0.2, -0.3, 0.4], dtype=np.float32))
    assert np.allclose(cleared[3:], np.zeros(3, dtype=np.float32))


def test_pid_scan_reverses_at_yaw_limits() -> None:
    layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = PidAimController(
        PidAimConfig(
            open_loop=OpenLoopAimConfig(yaw_step_rad=0.08, pitch_step_rad=0.08, action_layout=layout),
        )
    )

    action_pos, _ = controller.update({"qpos": [0.0, 0.0, 0.0, 1.6, 0.0]}, (480, 640, 3), dt=0.01)
    action_neg, _ = controller.update({"qpos": [0.0, 0.0, 0.0, -1.6, 0.0]}, (480, 640, 3), dt=0.01)

    assert action_pos[3] < 0.0
    assert action_neg[3] > 0.0


def test_pid_controller_hands_off_from_scan_to_tracking() -> None:
    layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = PidAimController(
        PidAimConfig(
            open_loop=OpenLoopAimConfig(yaw_step_rad=0.08, pitch_step_rad=0.08, action_layout=layout),
        )
    )

    scan_action, scan_metrics = controller.update({"qpos": [0.0, 0.0, 0.0, 0.2, 0.0]}, (480, 640, 3), dt=0.01)
    track_action, track_metrics = controller.update(_info(), (480, 640, 3), dt=0.01)

    assert scan_metrics.as_dict()["scan_yaw_command"] == scan_action[3]
    assert "plane_x" in track_metrics.as_dict()
    assert track_action[3] < 0.0
    assert track_action[3] != scan_action[3]


def test_resolve_tracking_info_replaces_oracle_with_vision_detection() -> None:
    class FakeDetector:
        def detect(self, frame_rgb: np.ndarray) -> BullseyeDetection | None:
            return BullseyeDetection(pixel_x=123.5, pixel_y=45.25, score=0.9, x_norm=0.2, y_norm=0.3)

    resolved = _resolve_tracking_info(
        {
            "bullseye_pixel": [480, 120],
            "camera_fovy_deg": 60.0,
            "camera_fovx_deg": 80.0,
        },
        np.zeros((480, 640, 3), dtype=np.uint8),
        bullseye_source=BullseyeSource.VISION,
        bullseye_detector=FakeDetector(),
    )

    assert resolved["bullseye_source"] == "vision"
    assert resolved["oracle_bullseye_pixel"] == [480.0, 120.0]
    assert resolved["bullseye_pixel"] == [123.5, 45.25]
    assert resolved["vision_bullseye_score"] == 0.9


def test_resolve_tracking_info_does_not_fallback_to_oracle_when_vision_misses() -> None:
    class FakeDetector:
        def detect(self, frame_rgb: np.ndarray) -> BullseyeDetection | None:
            return None

    resolved = _resolve_tracking_info(
        {
            "bullseye_pixel": [480, 120],
            "camera_fovy_deg": 60.0,
            "camera_fovx_deg": 80.0,
        },
        np.zeros((480, 640, 3), dtype=np.uint8),
        bullseye_source=BullseyeSource.VISION,
        bullseye_detector=FakeDetector(),
    )

    assert resolved["bullseye_source"] == "vision"
    assert resolved["oracle_bullseye_pixel"] == [480.0, 120.0]
    assert "bullseye_pixel" not in resolved
