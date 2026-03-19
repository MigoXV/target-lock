from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from target_lock.controllers import ActionLayout
from target_lock import runtime as common_module
from target_lock.runtime import AlignmentThreshold, run_session


class _AlignedMetrics:
    def as_dict(self) -> dict[str, float]:
        return {
            "plane_x": 0.0,
            "plane_y": 0.0,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
        }


class _AlwaysAlignedController:
    def reset(self) -> None:
        return None

    def update(
        self,
        info: dict[str, object],
        frame_shape: tuple[int, int, int],
        dt: float | None = None,
    ) -> tuple[np.ndarray, _AlignedMetrics] | None:
        del info, frame_shape, dt
        return np.zeros(6, dtype=np.float32), _AlignedMetrics()


def test_run_session_fires_every_aligned_step(monkeypatch) -> None:
    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    recorded_actions: list[np.ndarray] = []

    class FakeSession:
        def __init__(self, server_addr: str) -> None:
            self.server_addr = server_addr

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def reset(self) -> np.ndarray:
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def step(self, action: np.ndarray):
            recorded_actions.append(action.copy())
            return SimpleNamespace(
                observation=np.zeros((8, 8, 3), dtype=np.uint8),
                info={
                    "camera_fovy_deg": 60.0,
                    "camera_fovx_deg": 80.0,
                    "width": 8,
                    "height": 8,
                    "fire": {},
                },
                reward=0.0,
                terminated=False,
                truncated=False,
            )

        def decode_frame(self, observation, info: dict[str, object]) -> np.ndarray:
            del observation, info
            return np.zeros((8, 8, 3), dtype=np.uint8)

    fake_cv2 = SimpleNamespace(
        WINDOW_NORMAL=0,
        namedWindow=lambda *args, **kwargs: None,
        imshow=lambda *args, **kwargs: None,
        waitKey=lambda *args, **kwargs: -1,
        destroyAllWindows=lambda: None,
        cvtColor=lambda frame, code: frame,
        COLOR_RGB2BGR=0,
        line=lambda *args, **kwargs: None,
        circle=lambda *args, **kwargs: None,
        putText=lambda *args, **kwargs: None,
        rectangle=lambda *args, **kwargs: None,
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=0,
    )

    monkeypatch.setattr(common_module, "LockonSession", FakeSession)
    monkeypatch.setattr(common_module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    run_session(
        server_addr="127.0.0.1:50051",
        controller=_AlwaysAlignedController(),
        action_layout=action_layout,
        max_steps=3,
        threshold=AlignmentThreshold(azimuth_deg=0.1, elevation_deg=0.1, plane_x=0.01, plane_y=0.01),
        fire_when_aligned=True,
    )

    fire_actions = [action for action in recorded_actions if action[5] == 1.0]
    assert len(fire_actions) == 3
