from __future__ import annotations

import math

import numpy as np

from target_lock.geometry import backproject_direction, backproject_to_spherical


def test_backproject_direction_center_is_forward() -> None:
    direction = backproject_direction((0.0, 0.0), camera_fovy_deg=60.0, camera_fovx_deg=80.0)
    assert np.allclose(direction, np.array([1.0, 0.0, 0.0]))


def test_backproject_to_spherical_right_up_quadrant() -> None:
    spherical = backproject_to_spherical((0.5, 0.5), camera_fovy_deg=60.0, camera_fovx_deg=80.0)
    assert spherical.azimuth_rad > 0.0
    assert spherical.elevation_rad > 0.0
    assert math.isclose(np.linalg.norm(spherical.vector), 1.0)
