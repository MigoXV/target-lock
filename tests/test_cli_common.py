from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np

from target_lock.controllers import ActionLayout
import target_lock.runner.runner as common_module
from target_lock.runner import AlignmentThreshold, BullseyeSource, Runner
from target_lock.vision import AsyncCvBullseyeVision, BullseyeDetection


class _AlignedMetrics:
    def as_dict(self) -> dict[str, float]:
        return {
            "plane_x": 0.0,
            "plane_y": 0.0,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
        }


class _AlwaysAlignedController:
    def __init__(self) -> None:
        self.seen_info: list[dict[str, object]] = []

    def reset(self) -> None:
        return None

    def update(
        self,
        info: dict[str, object],
        frame_shape: tuple[int, int, int],
        dt: float | None = None,
    ) -> tuple[np.ndarray, _AlignedMetrics] | None:
        del frame_shape, dt
        self.seen_info.append(dict(info))
        return np.zeros(6, dtype=np.float32), _AlignedMetrics()


def _fake_cv2() -> SimpleNamespace:
    return SimpleNamespace(
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


def test_runner_fires_every_aligned_step(monkeypatch) -> None:
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
            return np.zeros((8, 40, 3), dtype=np.uint8)

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

    monkeypatch.setattr(common_module, "LockonSession", FakeSession)
    monkeypatch.setattr(common_module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())

    Runner(
        server_addr="127.0.0.1:50051",
        controller=_AlwaysAlignedController(),
        action_layout=action_layout,
        max_steps=3,
        threshold=AlignmentThreshold(azimuth_deg=0.1, elevation_deg=0.1, plane_x=0.01, plane_y=0.01),
        fire_when_aligned=True,
    ).run()

    fire_actions = [action for action in recorded_actions if action[5] == 1.0]
    assert len(fire_actions) == 3


def test_runner_uses_vision_detection_when_available(monkeypatch) -> None:
    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)

    class FakeSession:
        def __init__(self, server_addr: str) -> None:
            self.server_addr = server_addr

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def reset(self) -> np.ndarray:
            return np.zeros((8, 40, 3), dtype=np.uint8)

        def step(self, action: np.ndarray):
            del action
            return SimpleNamespace(
                observation=np.zeros((8, 8, 3), dtype=np.uint8),
                info={
                    "bullseye_pixel": [4, 2],
                    "camera_fovy_deg": 60.0,
                    "camera_fovx_deg": 80.0,
                    "width": 40,
                    "height": 8,
                    "fire": {},
                },
                reward=0.0,
                terminated=False,
                truncated=False,
            )

        def decode_frame(self, observation, info: dict[str, object]) -> np.ndarray:
            del observation, info
            return np.zeros((8, 40, 3), dtype=np.uint8)

    class FakeDetector:
        def detect(self, frame_rgb: np.ndarray) -> BullseyeDetection | None:
            del frame_rgb
            return BullseyeDetection(pixel_x=3.5, pixel_y=1.25, score=0.9, x_norm=0.2, y_norm=0.3)

    monkeypatch.setattr(common_module, "LockonSession", FakeSession)
    monkeypatch.setattr(common_module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())

    result = Runner(
        server_addr="127.0.0.1:50051",
        controller=_AlwaysAlignedController(),
        action_layout=action_layout,
        max_steps=1,
        threshold=AlignmentThreshold(azimuth_deg=0.1, elevation_deg=0.1, plane_x=0.01, plane_y=0.01),
        fire_when_aligned=False,
        bullseye_source=BullseyeSource.VISION,
        bullseye_detector=FakeDetector(),
    ).run()

    last_info = result["last_info"]
    assert last_info["bullseye_source"] == "vision"
    assert last_info["oracle_bullseye_pixel"] == [4.0, 2.0]
    assert last_info["bullseye_pixel"] == [3.5, 1.25]
    assert last_info["vision_bullseye_score"] == 0.9


def test_runner_does_not_fallback_to_oracle_when_vision_misses(monkeypatch) -> None:
    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)

    class FakeSession:
        def __init__(self, server_addr: str) -> None:
            self.server_addr = server_addr

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def reset(self) -> np.ndarray:
            return np.zeros((8, 40, 3), dtype=np.uint8)

        def step(self, action: np.ndarray):
            del action
            return SimpleNamespace(
                observation=np.zeros((8, 40, 3), dtype=np.uint8),
                info={
                    "bullseye_pixel": [4, 2],
                    "camera_fovy_deg": 60.0,
                    "camera_fovx_deg": 80.0,
                    "width": 40,
                    "height": 8,
                    "fire": {},
                },
                reward=0.0,
                terminated=False,
                truncated=False,
            )

        def decode_frame(self, observation, info: dict[str, object]) -> np.ndarray:
            del observation, info
            return np.zeros((8, 40, 3), dtype=np.uint8)

    class FakeDetector:
        def detect(self, frame_rgb: np.ndarray) -> BullseyeDetection | None:
            del frame_rgb
            return None

    monkeypatch.setattr(common_module, "LockonSession", FakeSession)
    monkeypatch.setattr(common_module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())

    result = Runner(
        server_addr="127.0.0.1:50051",
        controller=_AlwaysAlignedController(),
        action_layout=action_layout,
        max_steps=1,
        threshold=AlignmentThreshold(azimuth_deg=0.1, elevation_deg=0.1, plane_x=0.01, plane_y=0.01),
        fire_when_aligned=False,
        bullseye_source=BullseyeSource.VISION,
        bullseye_detector=FakeDetector(),
    ).run()

    last_info = result["last_info"]
    assert last_info["bullseye_source"] == "vision"
    assert last_info["oracle_bullseye_pixel"] == [4.0, 2.0]
    assert "bullseye_pixel" not in last_info


def test_runner_downsamples_vision_detection_and_reuses_last_result(monkeypatch) -> None:
    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = _AlwaysAlignedController()

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
            del action
            return SimpleNamespace(
                observation=np.zeros((8, 40, 3), dtype=np.uint8),
                info={
                    "bullseye_pixel": [4, 2],
                    "camera_fovy_deg": 60.0,
                    "camera_fovx_deg": 80.0,
                    "width": 40,
                    "height": 8,
                    "fire": {},
                },
                reward=0.0,
                terminated=False,
                truncated=False,
            )

        def decode_frame(self, observation, info: dict[str, object]) -> np.ndarray:
            del observation, info
            return np.zeros((8, 40, 3), dtype=np.uint8)

    class FakeDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect(self, frame_rgb: np.ndarray) -> BullseyeDetection | None:
            del frame_rgb
            self.calls += 1
            pixel_x = 10.0 + self.calls
            return BullseyeDetection(pixel_x=pixel_x, pixel_y=1.0, score=0.9, x_norm=0.2, y_norm=0.3)

    detector = FakeDetector()
    monkeypatch.setattr(common_module, "LockonSession", FakeSession)
    monkeypatch.setattr(common_module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())

    Runner(
        server_addr="127.0.0.1:50051",
        controller=controller,
        action_layout=action_layout,
        max_steps=3,
        threshold=AlignmentThreshold(azimuth_deg=0.1, elevation_deg=0.1, plane_x=0.01, plane_y=0.01),
        fire_when_aligned=False,
        bullseye_source=BullseyeSource.VISION,
        bullseye_detector=detector,
        vision_detect_every_n_frames=2,
    ).run()

    assert detector.calls == 2
    assert controller.seen_info[0]["bullseye_pixel"] == [11.0, 1.0]
    assert controller.seen_info[1]["bullseye_pixel"] == [11.0, 1.0]
    assert controller.seen_info[2]["bullseye_pixel"] == [12.0, 1.0]


def test_runner_smooths_vision_detection(monkeypatch) -> None:
    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = _AlwaysAlignedController()

    class FakeSession:
        def __init__(self, server_addr: str) -> None:
            self.server_addr = server_addr

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def reset(self) -> np.ndarray:
            return np.zeros((8, 40, 3), dtype=np.uint8)

        def step(self, action: np.ndarray):
            del action
            return SimpleNamespace(
                observation=np.zeros((8, 40, 3), dtype=np.uint8),
                info={
                    "bullseye_pixel": [4, 2],
                    "camera_fovy_deg": 60.0,
                    "camera_fovx_deg": 80.0,
                    "width": 40,
                    "height": 8,
                    "fire": {},
                },
                reward=0.0,
                terminated=False,
                truncated=False,
            )

        def decode_frame(self, observation, info: dict[str, object]) -> np.ndarray:
            del observation, info
            return np.zeros((8, 40, 3), dtype=np.uint8)

    class FakeDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect(self, frame_rgb: np.ndarray) -> BullseyeDetection | None:
            del frame_rgb
            self.calls += 1
            pixel_x = 10.0 if self.calls == 1 else 18.0
            return BullseyeDetection(pixel_x=pixel_x, pixel_y=2.0, score=0.9, x_norm=0.2, y_norm=0.3)

    detector = FakeDetector()
    monkeypatch.setattr(common_module, "LockonSession", FakeSession)
    monkeypatch.setattr(common_module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())

    Runner(
        server_addr="127.0.0.1:50051",
        controller=controller,
        action_layout=action_layout,
        max_steps=2,
        threshold=AlignmentThreshold(azimuth_deg=0.1, elevation_deg=0.1, plane_x=0.01, plane_y=0.01),
        fire_when_aligned=False,
        bullseye_source=BullseyeSource.VISION,
        bullseye_detector=detector,
        vision_smoothing_alpha=0.25,
    ).run()

    assert detector.calls == 2
    assert controller.seen_info[0]["bullseye_pixel"] == [10.0, 2.0]
    assert controller.seen_info[1]["bullseye_pixel"] == [12.0, 2.0]


def test_runner_polls_async_detector_between_submit_frames(monkeypatch) -> None:
    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = _AlwaysAlignedController()

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
            del action
            return SimpleNamespace(
                observation=np.zeros((8, 8, 3), dtype=np.uint8),
                info={
                    "bullseye_pixel": [4, 2],
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

    class FakeAsyncDetector:
        def __init__(self) -> None:
            self.calls = 0
            self.latest_detection: BullseyeDetection | None = None
            self.reset_called = False
            self.close_called = False

        def reset(self) -> None:
            self.reset_called = True
            self.latest_detection = None

        def detect(self, frame_rgb: np.ndarray, info=None) -> BullseyeDetection | None:
            del frame_rgb, info
            self.calls += 1
            if self.calls == 1:
                self.latest_detection = BullseyeDetection(pixel_x=9.0, pixel_y=1.0, score=0.9, x_norm=0.2, y_norm=0.3)
                return None
            self.latest_detection = BullseyeDetection(pixel_x=15.0, pixel_y=1.0, score=0.9, x_norm=0.2, y_norm=0.3)
            return BullseyeDetection(pixel_x=9.0, pixel_y=1.0, score=0.9, x_norm=0.2, y_norm=0.3)

        def get_latest_detection(self) -> BullseyeDetection | None:
            return self.latest_detection

        def close(self) -> None:
            self.close_called = True

    detector = FakeAsyncDetector()
    monkeypatch.setattr(common_module, "LockonSession", FakeSession)
    monkeypatch.setattr(common_module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())

    Runner(
        server_addr="127.0.0.1:50051",
        controller=controller,
        action_layout=action_layout,
        max_steps=4,
        threshold=AlignmentThreshold(azimuth_deg=0.1, elevation_deg=0.1, plane_x=0.01, plane_y=0.01),
        fire_when_aligned=False,
        bullseye_source=BullseyeSource.VISION,
        bullseye_detector=detector,
        vision_detect_every_n_frames=2,
    ).run()

    assert detector.reset_called is True
    assert detector.close_called is True
    assert detector.calls == 2
    assert "bullseye_pixel" not in controller.seen_info[0]
    assert controller.seen_info[1]["bullseye_pixel"] == [9.0, 1.0]
    assert controller.seen_info[2]["bullseye_pixel"] == [9.0, 1.0]
    assert controller.seen_info[3]["bullseye_pixel"] == [15.0, 1.0]


def test_runner_with_async_cv_detector_does_not_block_main_loop(monkeypatch) -> None:
    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    controller = _AlwaysAlignedController()

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
            del action
            return SimpleNamespace(
                observation=np.zeros((8, 8, 3), dtype=np.uint8),
                info={
                    "bullseye_pixel": [4, 2],
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

    class SlowDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect(self, frame_rgb: np.ndarray, info=None) -> BullseyeDetection | None:
            del frame_rgb, info
            self.calls += 1
            time.sleep(0.05)
            return BullseyeDetection(pixel_x=7.0, pixel_y=2.0, score=0.9, x_norm=0.2, y_norm=0.3)

    slow_detector = SlowDetector()
    async_detector = AsyncCvBullseyeVision(
        onnx_path="unused.onnx",
        detector_factory=lambda: slow_detector,
    )
    monkeypatch.setattr(common_module, "LockonSession", FakeSession)
    monkeypatch.setattr(common_module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2())

    result = Runner(
        server_addr="127.0.0.1:50051",
        controller=controller,
        action_layout=action_layout,
        max_steps=5,
        threshold=AlignmentThreshold(azimuth_deg=0.1, elevation_deg=0.1, plane_x=0.01, plane_y=0.01),
        fire_when_aligned=False,
        bullseye_source=BullseyeSource.VISION,
        bullseye_detector=async_detector,
    ).run()

    assert len(controller.seen_info) == 5
    assert slow_detector.calls < len(controller.seen_info)
    assert "bullseye_pixel" not in controller.seen_info[0]


def test_runner_forwards_keyboard_input_to_action_mutator(monkeypatch) -> None:
    action_layout = ActionLayout(size=6, yaw_index=3, pitch_index=4, fire_index=5)
    seen_keys: list[int] = []

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
            del action
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

    class FakeMutator:
        def __call__(self, step_idx: int, action: np.ndarray) -> np.ndarray:
            del step_idx
            return action

        def handle_key(self, key: int) -> None:
            seen_keys.append(key)

    key_codes = iter([ord("w"), 27])
    fake_cv2 = _fake_cv2()
    fake_cv2.waitKey = lambda *args, **kwargs: next(key_codes)

    monkeypatch.setattr(common_module, "LockonSession", FakeSession)
    monkeypatch.setattr(common_module.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    Runner(
        server_addr="127.0.0.1:50051",
        controller=_AlwaysAlignedController(),
        action_layout=action_layout,
        max_steps=5,
        threshold=AlignmentThreshold(azimuth_deg=0.1, elevation_deg=0.1, plane_x=0.01, plane_y=0.01),
        fire_when_aligned=False,
        action_mutator=FakeMutator(),
    ).run()

    assert seen_keys == [ord("w"), 27]
