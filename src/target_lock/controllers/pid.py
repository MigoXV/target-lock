from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from target_lock.controllers.base import AimController, AimMetrics
from target_lock.controllers.open_loop import OpenLoopAimConfig, OpenLoopAimController, normalize_plane_coordinate
from target_lock.geometry import backproject_to_spherical


@dataclass
class AxisPid:
    kp: float
    ki: float
    kd: float
    integral_limit: float
    output_limit: float
    deadband: float = 0.0
    integral: float = 0.0
    prev_error: float = 0.0
    initialized: bool = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error: float, dt: float) -> tuple[float, dict[str, float]]:
        if abs(error) < self.deadband:
            error = 0.0

        derivative = 0.0 if not self.initialized else (error - self.prev_error) / max(dt, 1e-6)
        self.initialized = True
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -self.integral_limit, self.integral_limit))
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = float(np.clip(output, -self.output_limit, self.output_limit))
        self.prev_error = error
        return output, {
            "p": self.kp * error,
            "i": self.ki * self.integral,
            "d": self.kd * derivative,
            "output": output,
        }


@dataclass(frozen=True, slots=True)
class PidAimConfig:
    open_loop: OpenLoopAimConfig
    ff_gain: float = 0.35
    yaw_kp: float = 1.1
    yaw_ki: float = 0.08
    yaw_kd: float = 0.18
    pitch_kp: float = 1.1
    pitch_ki: float = 0.08
    pitch_kd: float = 0.18
    pid_deadband: float = 0.002
    integral_limit: float = 0.4
    feedback_limit: float = 0.7


@dataclass(frozen=True, slots=True)
class PidAimMetrics(AimMetrics):
    plane_x: float
    plane_y: float
    azimuth_deg: float
    elevation_deg: float
    yaw_ff: float
    pitch_ff: float
    yaw_fb: float
    pitch_fb: float
    yaw_p: float
    yaw_i: float
    yaw_d: float
    pitch_p: float
    pitch_i: float
    pitch_d: float

    def as_dict(self) -> dict[str, float]:
        return {
            "plane_x": self.plane_x,
            "plane_y": self.plane_y,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "yaw_ff": self.yaw_ff,
            "pitch_ff": self.pitch_ff,
            "yaw_fb": self.yaw_fb,
            "pitch_fb": self.pitch_fb,
            "yaw_p": self.yaw_p,
            "yaw_i": self.yaw_i,
            "yaw_d": self.yaw_d,
            "pitch_p": self.pitch_p,
            "pitch_i": self.pitch_i,
            "pitch_d": self.pitch_d,
        }


class PidAimController(AimController):
    def __init__(self, config: PidAimConfig) -> None:
        self.config = config
        self.open_loop = OpenLoopAimController(config.open_loop)
        self.yaw_pid = AxisPid(
            kp=config.yaw_kp,
            ki=config.yaw_ki,
            kd=config.yaw_kd,
            integral_limit=config.integral_limit,
            output_limit=config.feedback_limit,
            deadband=config.pid_deadband,
        )
        self.pitch_pid = AxisPid(
            kp=config.pitch_kp,
            ki=config.pitch_ki,
            kd=config.pitch_kd,
            integral_limit=config.integral_limit,
            output_limit=config.feedback_limit,
            deadband=config.pid_deadband,
        )

    def reset(self) -> None:
        self.yaw_pid.reset()
        self.pitch_pid.reset()

    def update(
        self,
        info: dict[str, Any],
        frame_shape: tuple[int, int, int],
        dt: float | None = None,
    ) -> tuple[np.ndarray, PidAimMetrics] | None:
        if dt is None:
            raise ValueError("dt is required for PID controller")

        bullseye_pixel = info.get("bullseye_pixel")
        if not isinstance(bullseye_pixel, list) or len(bullseye_pixel) != 2:
            self.reset()
            return None

        width = int(info.get("width", frame_shape[1]))
        height = int(info.get("height", frame_shape[0]))
        plane_x, plane_y = normalize_plane_coordinate(bullseye_pixel, width=width, height=height)
        spherical = backproject_to_spherical(
            (plane_x, plane_y),
            camera_fovy_deg=float(info["camera_fovy_deg"]),
            camera_fovx_deg=float(info["camera_fovx_deg"]),
        )

        yaw_ff = self.config.ff_gain * (spherical.azimuth_rad / self.config.open_loop.yaw_step_rad)
        pitch_ff = self.config.ff_gain * (spherical.elevation_rad / self.config.open_loop.pitch_step_rad)
        yaw_fb, yaw_terms = self.yaw_pid.update(plane_x, dt)
        pitch_fb, pitch_terms = self.pitch_pid.update(plane_y, dt)

        action = self.config.open_loop.action_layout.build_idle()
        action[self.config.open_loop.action_layout.yaw_index] = np.clip(yaw_ff + yaw_fb, -1.0, 1.0)
        action[self.config.open_loop.action_layout.pitch_index] = np.clip(pitch_ff + pitch_fb, -1.0, 1.0)
        return action, PidAimMetrics(
            plane_x=plane_x,
            plane_y=plane_y,
            azimuth_deg=spherical.azimuth_deg,
            elevation_deg=spherical.elevation_deg,
            yaw_ff=float(yaw_ff),
            pitch_ff=float(pitch_ff),
            yaw_fb=yaw_fb,
            pitch_fb=pitch_fb,
            yaw_p=yaw_terms["p"],
            yaw_i=yaw_terms["i"],
            yaw_d=yaw_terms["d"],
            pitch_p=pitch_terms["p"],
            pitch_i=pitch_terms["i"],
            pitch_d=pitch_terms["d"],
        )
