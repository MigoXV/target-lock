from __future__ import annotations

import argparse

from target_lock.cli.common import AlignmentThreshold, run_session
from target_lock.controllers import ActionLayout, OpenLoopAimConfig, OpenLoopAimController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-addr", default="127.0.0.1:50051")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--align-threshold-deg", type=float, default=0.25)
    parser.add_argument("--yaw-step-rad", type=float, default=0.08)
    parser.add_argument("--pitch-step-rad", type=float, default=0.08)
    parser.add_argument("--fire-when-aligned", action="store_true", default=True)
    parser.add_argument("--no-fire", action="store_false", dest="fire_when_aligned")
    args = parser.parse_args()

    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = OpenLoopAimController(
        OpenLoopAimConfig(
            yaw_step_rad=args.yaw_step_rad,
            pitch_step_rad=args.pitch_step_rad,
            action_layout=action_layout,
        )
    )
    run_session(
        server_addr=args.server_addr,
        controller=controller,
        action_layout=action_layout,
        max_steps=args.max_steps,
        threshold=AlignmentThreshold(
            azimuth_deg=args.align_threshold_deg,
            elevation_deg=args.align_threshold_deg,
        ),
        fire_when_aligned=args.fire_when_aligned,
    )


if __name__ == "__main__":
    main()
