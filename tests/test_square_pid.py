from __future__ import annotations

import numpy as np

from target_lock.cli.square_pid import _square_trajectory_action


def test_square_trajectory_cycles_four_edges() -> None:
    action = np.zeros(6, dtype=np.float32)
    a0 = _square_trajectory_action(0, action.copy(), segment_steps=10, move_speed=0.3, base_rot=0.0)
    a1 = _square_trajectory_action(10, action.copy(), segment_steps=10, move_speed=0.3, base_rot=0.0)
    a2 = _square_trajectory_action(20, action.copy(), segment_steps=10, move_speed=0.3, base_rot=0.0)
    a3 = _square_trajectory_action(30, action.copy(), segment_steps=10, move_speed=0.3, base_rot=0.0)

    assert np.allclose(a0[:2], np.array([0.3, 0.0], dtype=np.float32))
    assert np.allclose(a1[:2], np.array([0.0, 0.3], dtype=np.float32))
    assert np.allclose(a2[:2], np.array([-0.3, 0.0], dtype=np.float32))
    assert np.allclose(a3[:2], np.array([0.0, -0.3], dtype=np.float32))
