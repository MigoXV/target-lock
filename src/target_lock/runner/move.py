from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from target_lock.commands.config import (
    AlignmentConfig,
    ControllerConfig,
    MotionConfig,
    MoveCommandConfig,
    TrackingConfig,
    VisionConfig,
)
from target_lock.controllers import ActionLayout, OpenLoopAimConfig, PidAimConfig, PidAimController
from target_lock.runner.runner import AlignmentThreshold, BullseyeSource, Runner
from target_lock.vision import AsyncCvBullseyeVision, CvBullseyeVision, OracleBullseyeVision, resolve_autoaim_onnx_path


MANUAL_BASE_KEY_LATCH_SECONDS = 0.12


def build_action_layout() -> ActionLayout:
    return ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)


def build_alignment_threshold(config: AlignmentConfig) -> AlignmentThreshold:
    return AlignmentThreshold(
        azimuth_deg=config.azimuth_deg,
        elevation_deg=config.elevation_deg,
        plane_x=config.plane_x,
        plane_y=config.plane_y,
    )


def build_pid_controller(
    *,
    action_layout: ActionLayout,
    config: ControllerConfig,
) -> PidAimController:
    return PidAimController(
        PidAimConfig(
            open_loop=OpenLoopAimConfig(
                yaw_step_rad=config.open_loop.yaw_step_rad,
                pitch_step_rad=config.open_loop.pitch_step_rad,
                action_layout=action_layout,
            ),
            scan_yaw_command=config.scan_yaw_command,
            scan_limit_rad=np.deg2rad(config.scan_limit_deg),
            yaw_kp=config.pid.yaw.kp,
            yaw_ki=config.pid.yaw.ki,
            yaw_kd=config.pid.yaw.kd,
            pitch_kp=config.pid.pitch.kp,
            pitch_ki=config.pid.pitch.ki,
            pitch_kd=config.pid.pitch.kd,
            pid_deadband=config.pid.deadband,
            integral_limit=config.pid.integral_limit,
            feedback_limit=config.pid.feedback_limit,
        )
    )


def build_bullseye_detector(
    *,
    tracking: TrackingConfig,
    vision: VisionConfig,
):
    bullseye_source = BullseyeSource(tracking.bullseye_source)
    if bullseye_source == BullseyeSource.ORACLE:
        return OracleBullseyeVision()
    detector_cls = AsyncCvBullseyeVision if vision.async_inference else CvBullseyeVision
    detector_kwargs = {
        "onnx_path": resolve_autoaim_onnx_path(None, vision.onnx_path),
        "img_size_fallback": vision.img_size_fallback,
        "score_threshold": vision.score_threshold,
    }
    if vision.async_inference:
        detector_kwargs["smoothing_alpha"] = vision.smoothing_alpha
    return detector_cls(**detector_kwargs)


def random_trajectory_action(
    step_idx: int,
    current_motion: np.ndarray,
    *,
    rng: np.random.Generator,
    hold_steps: int,
    move_speed: float,
    base_rot_scale: float,
) -> np.ndarray:
    if step_idx % max(hold_steps, 1) == 0:
        # Keep both planar axes active while treating move_speed as the total speed cap.
        min_angle = np.deg2rad(20.0)
        max_angle = np.deg2rad(70.0)
        angle = float(rng.uniform(min_angle, max_angle))
        speed = float(rng.uniform(0.7 * move_speed, move_speed))
        move_x = speed * np.cos(angle) * float(rng.choice((-1.0, 1.0)))
        move_y = speed * np.sin(angle) * float(rng.choice((-1.0, 1.0)))
        current_motion = np.array(
            [
                move_x,
                move_y,
                rng.uniform(-base_rot_scale, base_rot_scale),
            ],
            dtype=np.float32,
        )
    return current_motion


def square_trajectory_action(
    step_idx: int,
    action: np.ndarray,
    *,
    segment_steps: int,
    move_speed: float,
    base_rot: float,
) -> np.ndarray:
    directions = (
        (move_speed, 0.0),
        (0.0, move_speed),
        (-move_speed, 0.0),
        (0.0, -move_speed),
    )
    segment_index = (step_idx // max(segment_steps, 1)) % len(directions)
    move_x, move_y = directions[segment_index]
    action[0] = move_x
    action[1] = move_y
    action[2] = base_rot
    return action


@dataclass(slots=True)
class ManualBaseMotionMutator:
    move_speed: float
    base_rot_scale: float
    key_latch_seconds: float = MANUAL_BASE_KEY_LATCH_SECONDS
    _last_command: str | None = field(init=False, default=None)
    _last_command_at: float = field(init=False, default=0.0)

    def handle_key(self, key_code: int) -> None:
        if key_code < 0:
            return
        try:
            key = chr(key_code & 0xFF).lower()
        except ValueError:
            return

        if key == " ":
            self._last_command = None
            self._last_command_at = 0.0
            return

        if key not in {"w", "a", "s", "d", "q", "e"}:
            return

        self._last_command = key
        self._last_command_at = time.monotonic()

    def __call__(self, step_idx: int, action: np.ndarray) -> np.ndarray:
        del step_idx
        action[0] = 0.0
        action[1] = 0.0
        action[2] = 0.0

        if self._last_command is None:
            return action
        if time.monotonic() - self._last_command_at > self.key_latch_seconds:
            self._last_command = None
            return action

        if self._last_command == "w":
            action[0] = self.move_speed
        elif self._last_command == "s":
            action[0] = -self.move_speed
        elif self._last_command == "a":
            action[1] = self.move_speed
        elif self._last_command == "d":
            action[1] = -self.move_speed
        elif self._last_command == "q":
            action[2] = self.base_rot_scale
        elif self._last_command == "e":
            action[2] = -self.base_rot_scale
        return action


def build_random_motion_mutator(
    config: MotionConfig,
) -> Callable[[int, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(config.seed)
    current_motion = np.zeros(3, dtype=np.float32)

    def action_mutator(step_idx: int, action: np.ndarray) -> np.ndarray:
        nonlocal current_motion
        current_motion = random_trajectory_action(
            step_idx,
            current_motion,
            rng=rng,
            hold_steps=config.hold_steps,
            move_speed=config.move_speed,
            base_rot_scale=config.base_rot_scale,
        )
        action[0] = current_motion[0]
        action[1] = current_motion[1]
        action[2] = current_motion[2]
        return action

    return action_mutator


def build_motion_mutator(
    config: MotionConfig,
) -> Callable[[int, np.ndarray], np.ndarray]:
    if config.manual_base_control:
        return ManualBaseMotionMutator(
            move_speed=config.move_speed,
            base_rot_scale=config.base_rot_scale,
        )
    return build_random_motion_mutator(config)


def run_move(config: MoveCommandConfig) -> dict[str, object]:
    bullseye_source = BullseyeSource(config.tracking.bullseye_source)
    action_layout = build_action_layout()
    controller = build_pid_controller(
        action_layout=action_layout,
        config=config.controller,
    )
    bullseye_detector = build_bullseye_detector(
        tracking=config.tracking,
        vision=config.vision,
    )
    use_async_vision = bullseye_source == BullseyeSource.VISION and config.vision.async_inference
    return Runner(
        server_addr=config.server.addr,
        controller=controller,
        action_layout=action_layout,
        max_steps=config.session.max_steps,
        threshold=build_alignment_threshold(config.tracking.alignment),
        fire_when_aligned=config.session.fire_when_aligned,
        action_mutator=build_motion_mutator(config.motion),
        bullseye_source=bullseye_source,
        bullseye_detector=bullseye_detector,
        vision_detect_every_n_frames=config.vision.detect_every_n_frames,
        vision_smoothing_alpha=1.0 if use_async_vision else config.vision.smoothing_alpha,
    ).run()


__all__ = [
    "build_action_layout",
    "build_alignment_threshold",
    "build_bullseye_detector",
    "build_motion_mutator",
    "build_pid_controller",
    "ManualBaseMotionMutator",
    "build_random_motion_mutator",
    "random_trajectory_action",
    "run_move",
    "square_trajectory_action",
]
