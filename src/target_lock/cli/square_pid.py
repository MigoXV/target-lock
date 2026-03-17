from __future__ import annotations

import argparse

import numpy as np

from target_lock.cli.common import AlignmentThreshold, run_session
from target_lock.controllers import ActionLayout, OpenLoopAimConfig, PidAimConfig, PidAimController


def _square_trajectory_action(
    step_idx: int,
    action: np.ndarray,
    *,
    segment_steps: int,
    move_speed: float,
    base_rot: float,
) -> np.ndarray:
    segment = (step_idx // max(segment_steps, 1)) % 4
    if segment == 0:
        move_x, move_y = move_speed, 0.0
    elif segment == 1:
        move_x, move_y = 0.0, move_speed
    elif segment == 2:
        move_x, move_y = -move_speed, 0.0
    else:
        move_x, move_y = 0.0, -move_speed

    action[0] = move_x
    action[1] = move_y
    action[2] = base_rot
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-addr", default="127.0.0.1:50051")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--align-threshold-deg", type=float, default=0.25)
    parser.add_argument("--plane-threshold", type=float, default=0.015)
    parser.add_argument("--yaw-step-rad", type=float, default=0.08)
    parser.add_argument("--pitch-step-rad", type=float, default=0.08)
    parser.add_argument("--segment-steps", type=int, default=100)
    parser.add_argument("--move-speed", type=float, default=0.35)
    parser.add_argument("--base-rot", type=float, default=0.0)
    parser.add_argument("--random-rot-scale", type=float, default=0.35)
    parser.add_argument("--random-rot-hold-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ff-gain", type=float, default=3.0)
    parser.add_argument("--yaw-kp", type=float, default=1.8)
    parser.add_argument("--yaw-ki", type=float, default=0.12)
    parser.add_argument("--yaw-kd", type=float, default=0.24)
    parser.add_argument("--pitch-kp", type=float, default=1.8)
    parser.add_argument("--pitch-ki", type=float, default=0.12)
    parser.add_argument("--pitch-kd", type=float, default=0.24)
    parser.add_argument("--pid-deadband", type=float, default=0.002)
    parser.add_argument("--integral-limit", type=float, default=0.4)
    parser.add_argument("--feedback-limit", type=float, default=0.7)
    parser.add_argument("--fire-when-aligned", action="store_true", default=True)
    parser.add_argument("--no-fire", action="store_false", dest="fire_when_aligned")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    current_base_rot = float(args.base_rot)
    target_base_rot = float(args.base_rot)

    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = PidAimController(
        PidAimConfig(
            open_loop=OpenLoopAimConfig(
                yaw_step_rad=args.yaw_step_rad,
                pitch_step_rad=args.pitch_step_rad,
                action_layout=action_layout,
            ),
            ff_gain=args.ff_gain,
            yaw_kp=args.yaw_kp,
            yaw_ki=args.yaw_ki,
            yaw_kd=args.yaw_kd,
            pitch_kp=args.pitch_kp,
            pitch_ki=args.pitch_ki,
            pitch_kd=args.pitch_kd,
            pid_deadband=args.pid_deadband,
            integral_limit=args.integral_limit,
            feedback_limit=args.feedback_limit,
        )
    )

    def action_mutator(step_idx: int, action: np.ndarray) -> np.ndarray:
        nonlocal current_base_rot, target_base_rot
        if args.random_rot_scale > 0.0 and step_idx % max(args.random_rot_hold_steps, 1) == 0:
            target_base_rot = float(rng.uniform(-args.random_rot_scale, args.random_rot_scale))
        current_base_rot = target_base_rot
        return _square_trajectory_action(
            step_idx,
            action,
            segment_steps=args.segment_steps,
            move_speed=args.move_speed,
            base_rot=current_base_rot,
        )

    run_session(
        server_addr=args.server_addr,
        controller=controller,
        action_layout=action_layout,
        max_steps=args.max_steps,
        threshold=AlignmentThreshold(
            azimuth_deg=args.align_threshold_deg,
            elevation_deg=args.align_threshold_deg,
            plane_x=args.plane_threshold,
            plane_y=args.plane_threshold,
        ),
        fire_when_aligned=args.fire_when_aligned,
        action_mutator=action_mutator,
    )


if __name__ == "__main__":
    main()
