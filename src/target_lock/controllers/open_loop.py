from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from target_lock.controllers.base import ActionLayout, AimController, AimMetrics
from target_lock.geometry import backproject_to_spherical


DEFAULT_ACTION_LAYOUT = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)


@dataclass(frozen=True, slots=True)
class OpenLoopAimConfig:
    yaw_step_rad: float
    pitch_step_rad: float
    action_layout: ActionLayout = DEFAULT_ACTION_LAYOUT
    clip_limit: float = 1.0


@dataclass(frozen=True, slots=True)
class OpenLoopMetrics(AimMetrics):
    plane_x: float
    plane_y: float
    azimuth_deg: float
    elevation_deg: float
    yaw_command: float
    pitch_command: float

    def as_dict(self) -> dict[str, float]:
        return {
            "plane_x": self.plane_x,
            "plane_y": self.plane_y,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "yaw_command": self.yaw_command,
            "pitch_command": self.pitch_command,
        }


def normalize_plane_coordinate(
    bullseye_pixel: list[object],
    width: int,
    height: int,
) -> tuple[float, float]:
    px = float(bullseye_pixel[0])
    py = float(bullseye_pixel[1])
    plane_x = (px - width / 2.0) / (width / 2.0)
    plane_y = (height / 2.0 - py) / (height / 2.0)
    return plane_x, plane_y


class OpenLoopAimController(AimController):
    def __init__(self, config: OpenLoopAimConfig) -> None:
        self.config = config

    def reset(self) -> None:
        return None

    def update(
        self,
        info: dict[str, Any],
        frame_shape: tuple[int, int, int],
        dt: float | None = None,
    ) -> tuple[np.ndarray, OpenLoopMetrics] | None:
        del dt
        bullseye_pixel = info.get("bullseye_pixel")
        if not isinstance(bullseye_pixel, list) or len(bullseye_pixel) != 2:
            return None

        width = int(info.get("width", frame_shape[1]))
        height = int(info.get("height", frame_shape[0]))
        plane_x, plane_y = normalize_plane_coordinate(bullseye_pixel, width=width, height=height)
        spherical = backproject_to_spherical(
            (plane_x, plane_y),
            camera_fovy_deg=float(info["camera_fovy_deg"]),
            camera_fovx_deg=float(info["camera_fovx_deg"]),
        )

        yaw_command = float(
            np.clip(-spherical.azimuth_rad / self.config.yaw_step_rad, -self.config.clip_limit, self.config.clip_limit)
        )
        pitch_command = float(
            np.clip(
                spherical.elevation_rad / self.config.pitch_step_rad,
                -self.config.clip_limit,
                self.config.clip_limit,
            )
        )

        action = self.config.action_layout.build_idle()
        action[self.config.action_layout.yaw_index] = yaw_command
        action[self.config.action_layout.pitch_index] = pitch_command
        return action, OpenLoopMetrics(
            plane_x=plane_x,
            plane_y=plane_y,
            azimuth_deg=spherical.azimuth_deg,
            elevation_deg=spherical.elevation_deg,
            yaw_command=yaw_command,
            pitch_command=pitch_command,
        )
