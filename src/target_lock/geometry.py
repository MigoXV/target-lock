from __future__ import annotations

from dataclasses import dataclass
from math import atan, atan2, degrees, radians, sqrt, tan
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class SphericalDirection:
    vector: np.ndarray
    azimuth_rad: float
    elevation_rad: float

    @property
    def azimuth_deg(self) -> float:
        return degrees(self.azimuth_rad)

    @property
    def elevation_deg(self) -> float:
        return degrees(self.elevation_rad)


def _parse_plane_coordinate(target_plane_xy: Iterable[float]) -> tuple[float, float]:
    plane = tuple(float(v) for v in target_plane_xy)
    if len(plane) != 2:
        raise ValueError("target_plane_xy must contain exactly two values")
    return plane[0], plane[1]


def _resolve_horizontal_fov(
    camera_fovy_deg: float,
    camera_fovx_deg: float | None,
    aspect_ratio: float | None,
) -> float:
    if camera_fovx_deg is not None:
        return float(camera_fovx_deg)
    if aspect_ratio is None or aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be positive when camera_fovx_deg is omitted")
    fovy_rad = radians(float(camera_fovy_deg))
    return degrees(2.0 * atan(aspect_ratio * tan(fovy_rad / 2.0)))


def backproject_direction(
    target_plane_xy: Iterable[float],
    camera_fovy_deg: float,
    camera_fovx_deg: float | None = None,
    *,
    aspect_ratio: float | None = None,
    image_y_down: bool = False,
) -> np.ndarray:
    plane_x, plane_y = _parse_plane_coordinate(target_plane_xy)
    if image_y_down:
        plane_y = -plane_y

    fovx_deg = _resolve_horizontal_fov(camera_fovy_deg, camera_fovx_deg, aspect_ratio)
    tan_half_fovx = tan(radians(fovx_deg) / 2.0)
    tan_half_fovy = tan(radians(float(camera_fovy_deg)) / 2.0)

    forward = 1.0
    right = plane_x * tan_half_fovx
    up = plane_y * tan_half_fovy
    norm = sqrt(forward * forward + right * right + up * up)
    if norm == 0.0:
        raise ValueError("back-projected direction has zero norm")

    return np.array([forward / norm, right / norm, up / norm], dtype=np.float64)


def direction_to_spherical(direction: Iterable[float]) -> SphericalDirection:
    vector = np.asarray(tuple(float(v) for v in direction), dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("direction must contain exactly three values")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")

    unit = vector / norm
    forward, right, up = unit.tolist()
    azimuth_rad = atan2(right, forward)
    elevation_rad = atan2(up, sqrt(forward * forward + right * right))
    return SphericalDirection(vector=unit, azimuth_rad=azimuth_rad, elevation_rad=elevation_rad)


def backproject_to_spherical(
    target_plane_xy: Iterable[float],
    camera_fovy_deg: float,
    camera_fovx_deg: float | None = None,
    *,
    aspect_ratio: float | None = None,
    image_y_down: bool = False,
) -> SphericalDirection:
    direction = backproject_direction(
        target_plane_xy=target_plane_xy,
        camera_fovy_deg=camera_fovy_deg,
        camera_fovx_deg=camera_fovx_deg,
        aspect_ratio=aspect_ratio,
        image_y_down=image_y_down,
    )
    return direction_to_spherical(direction)
