from __future__ import annotations

from typing import Annotated

import numpy as np
import typer

from target_lock.cli.common import AlignmentThreshold, run_session
from target_lock.controllers import (
    ActionLayout,
    OpenLoopAimConfig,
    OpenLoopAimController,
    PidAimConfig,
    PidAimController,
)


app = typer.Typer(
    help="Command-line entrypoints for target locking demos.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


ServerAddrOption = Annotated[
    str,
    typer.Option(help="gRPC server address for the lockon simulator."),
]
MaxStepsOption = Annotated[
    int,
    typer.Option(help="Maximum number of control loop steps to run."),
]
MoveMaxStepsOption = Annotated[
    int | None,
    typer.Option(help="Maximum number of control loop steps to run. Leave empty for endless mode."),
]
AlignThresholdOption = Annotated[
    float,
    typer.Option(help="Alignment threshold in degrees for azimuth and elevation."),
]
YawStepOption = Annotated[
    float,
    typer.Option(help="Open-loop yaw step in radians."),
]
PitchStepOption = Annotated[
    float,
    typer.Option(help="Open-loop pitch step in radians."),
]
FireOption = Annotated[
    bool,
    typer.Option("--fire/--no-fire", help="Fire whenever the target becomes aligned."),
]


def _build_action_layout() -> ActionLayout:
    return ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)


@app.command("static")
def static_open_loop(
    server_addr: ServerAddrOption = "127.0.0.1:50051",
    max_steps: MaxStepsOption = 1000,
    align_threshold_deg: AlignThresholdOption = 0.25,
    yaw_step_rad: YawStepOption = 0.08,
    pitch_step_rad: PitchStepOption = 0.08,
    fire_when_aligned: FireOption = True,
) -> None:
    action_layout = _build_action_layout()
    controller = OpenLoopAimController(
        OpenLoopAimConfig(
            yaw_step_rad=yaw_step_rad,
            pitch_step_rad=pitch_step_rad,
            action_layout=action_layout,
        )
    )
    run_session(
        server_addr=server_addr,
        controller=controller,
        action_layout=action_layout,
        max_steps=max_steps,
        threshold=AlignmentThreshold(
            azimuth_deg=align_threshold_deg,
            elevation_deg=align_threshold_deg,
        ),
        fire_when_aligned=fire_when_aligned,
    )


@app.command("move")
def move_open_loop(
    server_addr: ServerAddrOption = "127.0.0.1:50051",
    max_steps: MoveMaxStepsOption = None,
    align_threshold_deg: AlignThresholdOption = 0.25,
    yaw_step_rad: YawStepOption = 0.08,
    pitch_step_rad: PitchStepOption = 0.08,
    move_speed: Annotated[float, typer.Option(help="Maximum random planar speed magnitude.")] = 0.03,
    base_rot_scale: Annotated[float, typer.Option(help="Maximum random base rotation magnitude.")] = 0.15,
    hold_steps: Annotated[int, typer.Option(help="Steps to hold each random movement sample.")] = 60,
    seed: Annotated[int, typer.Option(help="Random seed for movement sampling.")] = 0,
    fire_when_aligned: FireOption = True,
) -> None:
    rng = np.random.default_rng(seed)
    current_motion = np.zeros(3, dtype=np.float32)

    action_layout = _build_action_layout()
    controller = OpenLoopAimController(
        OpenLoopAimConfig(
            yaw_step_rad=yaw_step_rad,
            pitch_step_rad=pitch_step_rad,
            action_layout=action_layout,
        )
    )

    def action_mutator(step_idx: int, action: np.ndarray) -> np.ndarray:
        nonlocal current_motion
        current_motion = _random_trajectory_action(
            step_idx,
            current_motion,
            rng=rng,
            hold_steps=hold_steps,
            move_speed=move_speed,
            base_rot_scale=base_rot_scale,
        )
        action[0] = current_motion[0]
        action[1] = current_motion[1]
        action[2] = current_motion[2]
        return action

    run_session(
        server_addr=server_addr,
        controller=controller,
        action_layout=action_layout,
        max_steps=max_steps,
        threshold=AlignmentThreshold(
            azimuth_deg=align_threshold_deg,
            elevation_deg=align_threshold_deg,
        ),
        fire_when_aligned=fire_when_aligned,
        action_mutator=action_mutator,
    )


def _random_trajectory_action(
    step_idx: int,
    current_motion: np.ndarray,
    *,
    rng: np.random.Generator,
    hold_steps: int,
    move_speed: float,
    base_rot_scale: float,
) -> np.ndarray:
    if step_idx % max(hold_steps, 1) == 0:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        speed = float(rng.uniform(0.0, move_speed))
        current_motion = np.array(
            [
                speed * np.cos(angle),
                speed * np.sin(angle),
                rng.uniform(-base_rot_scale, base_rot_scale),
            ],
            dtype=np.float32,
        )
    return current_motion


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


@app.command("square-pid")
def square_pid(
    server_addr: ServerAddrOption = "127.0.0.1:50051",
    max_steps: MaxStepsOption = 4000,
    align_threshold_deg: AlignThresholdOption = 0.25,
    plane_threshold: Annotated[float, typer.Option(help="Image plane alignment threshold.")] = 0.015,
    yaw_step_rad: YawStepOption = 0.08,
    pitch_step_rad: PitchStepOption = 0.08,
    segment_steps: Annotated[int, typer.Option(help="Steps to hold each square trajectory segment.")] = 100,
    move_speed: Annotated[float, typer.Option(help="Base speed used for the square trajectory.")] = 0.35,
    base_rot: Annotated[float, typer.Option(help="Base rotation command bias.")] = 0.0,
    random_rot_scale: Annotated[float, typer.Option(help="Random base rotation scale.")] = 0.35,
    random_rot_hold_steps: Annotated[int, typer.Option(help="Steps to hold each random rotation sample.")] = 60,
    seed: Annotated[int, typer.Option(help="Random seed for rotation sampling.")] = 0,
    yaw_kp: Annotated[float, typer.Option(help="Yaw PID proportional gain.")] = 1.8,
    yaw_ki: Annotated[float, typer.Option(help="Yaw PID integral gain.")] = 0.12,
    yaw_kd: Annotated[float, typer.Option(help="Yaw PID derivative gain.")] = 0.24,
    pitch_kp: Annotated[float, typer.Option(help="Pitch PID proportional gain.")] = 1.8,
    pitch_ki: Annotated[float, typer.Option(help="Pitch PID integral gain.")] = 0.12,
    pitch_kd: Annotated[float, typer.Option(help="Pitch PID derivative gain.")] = 0.24,
    pid_deadband: Annotated[float, typer.Option(help="PID deadband threshold.")] = 0.002,
    integral_limit: Annotated[float, typer.Option(help="Integral clamp limit.")] = 0.4,
    feedback_limit: Annotated[float, typer.Option(help="Feedback clamp limit.")] = 0.7,
    fire_when_aligned: FireOption = True,
) -> None:
    rng = np.random.default_rng(seed)
    current_base_rot = float(base_rot)
    target_base_rot = float(base_rot)

    action_layout = _build_action_layout()
    controller = PidAimController(
        PidAimConfig(
            open_loop=OpenLoopAimConfig(
                yaw_step_rad=yaw_step_rad,
                pitch_step_rad=pitch_step_rad,
                action_layout=action_layout,
            ),
            yaw_kp=yaw_kp,
            yaw_ki=yaw_ki,
            yaw_kd=yaw_kd,
            pitch_kp=pitch_kp,
            pitch_ki=pitch_ki,
            pitch_kd=pitch_kd,
            pid_deadband=pid_deadband,
            integral_limit=integral_limit,
            feedback_limit=feedback_limit,
        )
    )

    def action_mutator(step_idx: int, action: np.ndarray) -> np.ndarray:
        nonlocal current_base_rot, target_base_rot
        if random_rot_scale > 0.0 and step_idx % max(random_rot_hold_steps, 1) == 0:
            target_base_rot = float(rng.uniform(-random_rot_scale, random_rot_scale))
        current_base_rot = target_base_rot
        return _square_trajectory_action(
            step_idx,
            action,
            segment_steps=segment_steps,
            move_speed=move_speed,
            base_rot=current_base_rot,
        )

    run_session(
        server_addr=server_addr,
        controller=controller,
        action_layout=action_layout,
        max_steps=max_steps,
        threshold=AlignmentThreshold(
            azimuth_deg=align_threshold_deg,
            elevation_deg=align_threshold_deg,
            plane_x=plane_threshold,
            plane_y=plane_threshold,
        ),
        fire_when_aligned=fire_when_aligned,
        action_mutator=action_mutator,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
