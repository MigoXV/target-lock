from __future__ import annotations

import numpy as np

from target_lock.runner.move import ManualBaseMotionMutator, random_trajectory_action, square_trajectory_action


def test_square_trajectory_cycles_four_edges() -> None:
    action = np.zeros(6, dtype=np.float32)
    a0 = square_trajectory_action(0, action.copy(), segment_steps=10, move_speed=0.3, base_rot=0.0)
    a1 = square_trajectory_action(10, action.copy(), segment_steps=10, move_speed=0.3, base_rot=0.0)
    a2 = square_trajectory_action(20, action.copy(), segment_steps=10, move_speed=0.3, base_rot=0.0)
    a3 = square_trajectory_action(30, action.copy(), segment_steps=10, move_speed=0.3, base_rot=0.0)

    assert np.allclose(a0[:2], np.array([0.3, 0.0], dtype=np.float32))
    assert np.allclose(a1[:2], np.array([0.0, 0.3], dtype=np.float32))
    assert np.allclose(a2[:2], np.array([-0.3, 0.0], dtype=np.float32))
    assert np.allclose(a3[:2], np.array([0.0, -0.3], dtype=np.float32))


def test_random_trajectory_holds_sample_between_updates() -> None:
    rng = np.random.default_rng(7)
    current = np.zeros(3, dtype=np.float32)

    first = random_trajectory_action(
        0,
        current,
        rng=rng,
        hold_steps=5,
        move_speed=0.3,
        base_rot_scale=0.2,
    )
    second = random_trajectory_action(
        1,
        first,
        rng=rng,
        hold_steps=5,
        move_speed=0.3,
        base_rot_scale=0.2,
    )
    third = random_trajectory_action(
        5,
        second,
        rng=rng,
        hold_steps=5,
        move_speed=0.3,
        base_rot_scale=0.2,
    )

    assert np.allclose(first, second)
    assert not np.allclose(first, third)
    assert first[0] != 0.0
    assert first[1] != 0.0
    assert np.linalg.norm(first[:2]) <= 0.3 + 1e-6


def test_manual_base_motion_mutator_maps_wasd_qe_to_base_axes() -> None:
    mutator = ManualBaseMotionMutator(move_speed=0.2, base_rot_scale=0.05, key_latch_seconds=10.0)
    action = np.zeros(6, dtype=np.float32)

    mutator.handle_key(ord("w"))
    assert np.allclose(mutator(0, action.copy())[:3], np.array([0.2, 0.0, 0.0], dtype=np.float32))

    mutator.handle_key(ord("s"))
    assert np.allclose(mutator(0, action.copy())[:3], np.array([-0.2, 0.0, 0.0], dtype=np.float32))

    mutator.handle_key(ord("a"))
    assert np.allclose(mutator(0, action.copy())[:3], np.array([0.0, 0.2, 0.0], dtype=np.float32))

    mutator.handle_key(ord("d"))
    assert np.allclose(mutator(0, action.copy())[:3], np.array([0.0, -0.2, 0.0], dtype=np.float32))

    mutator.handle_key(ord("q"))
    assert np.allclose(mutator(0, action.copy())[:3], np.array([0.0, 0.0, 0.05], dtype=np.float32))

    mutator.handle_key(ord("e"))
    assert np.allclose(mutator(0, action.copy())[:3], np.array([0.0, 0.0, -0.05], dtype=np.float32))

    mutator.handle_key(ord(" "))
    assert np.allclose(mutator(0, action.copy())[:3], np.zeros(3, dtype=np.float32))
