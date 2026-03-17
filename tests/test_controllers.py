from __future__ import annotations

import numpy as np

from target_lock.cli.common import _clear_aim_action
from target_lock.controllers import ActionLayout, OpenLoopAimConfig, OpenLoopAimController, PidAimConfig, PidAimController


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
    assert action[3] > 0.0
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
    assert controller.update({}, (480, 640, 3), dt=0.01) is None
    assert controller.yaw_pid.initialized is False


def test_clear_aim_action_preserves_non_aim_axes() -> None:
    layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    action = np.array([0.2, -0.3, 0.4, 0.9, -0.8, 1.0], dtype=np.float32)

    cleared = _clear_aim_action(action.copy(), layout)

    assert np.allclose(cleared[:3], np.array([0.2, -0.3, 0.4], dtype=np.float32))
    assert np.allclose(cleared[3:], np.zeros(3, dtype=np.float32))
