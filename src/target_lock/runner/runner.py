from __future__ import annotations

import inspect
import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from target_lock.controllers import ActionLayout, AimController
from target_lock.sim import LockonSession
from target_lock.vision import BullseyeDetection, BullseyeDetector
from target_lock.vision.base import build_detection


CONTROL_DT = 0.01
FRAME_SKIP = 5
SCHEMATIC_TARGET_X = 0.5
SCHEMATIC_TARGET_Y = 0.0
SCHEMATIC_WORLD_X = (-1.0, 0.7)
SCHEMATIC_WORLD_Y = (-1.0, 1.0)


class BullseyeSource(str, Enum):
    ORACLE = "oracle"
    VISION = "vision"


@dataclass(frozen=True, slots=True)
class AlignmentThreshold:
    azimuth_deg: float
    elevation_deg: float
    plane_x: float | None = None
    plane_y: float | None = None


@dataclass(slots=True)
class Runner:
    server_addr: str
    controller: AimController
    action_layout: ActionLayout
    max_steps: int | None
    threshold: AlignmentThreshold
    fire_when_aligned: bool
    action_mutator: Callable[[int, np.ndarray], np.ndarray] | None = None
    bullseye_source: BullseyeSource = BullseyeSource.ORACLE
    bullseye_detector: BullseyeDetector | None = None
    vision_detect_every_n_frames: int = 1
    vision_smoothing_alpha: float = 1.0
    _vision_frame_index: int = field(init=False, default=0)
    _last_vision_detection: BullseyeDetection | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.vision_detect_every_n_frames < 1:
            raise ValueError("vision_detect_every_n_frames must be >= 1")
        if not 0.0 < self.vision_smoothing_alpha <= 1.0:
            raise ValueError("vision_smoothing_alpha must be in (0.0, 1.0]")

    def run(self) -> dict[str, object]:
        import cv2

        last_info: dict[str, object] = {}
        last_metrics: dict[str, float] | None = None
        self._vision_frame_index = 0
        self._last_vision_detection = None
        self._reset_detector()
        try:
            with LockonSession(server_addr=self.server_addr) as session:
                frame_rgb = session.reset()
                cv2.namedWindow("target-lock", cv2.WINDOW_NORMAL)
                action = self.action_layout.build_idle()
                step_indices = range(self.max_steps) if self.max_steps is not None else itertools.count()

                try:
                    for step_idx in step_indices:
                        if self.action_mutator is not None:
                            action = self.action_mutator(step_idx, action.copy())

                        step_result = session.step(action)
                        last_info = step_result.info
                        frame_rgb = session.decode_frame(step_result.observation, last_info)
                        last_info = self._resolve_tracking_info(last_info, frame_rgb)

                        computed = self.controller.update(last_info, frame_rgb.shape, dt=CONTROL_DT)
                        if self.action_mutator is None:
                            action = self.action_layout.build_idle()
                        else:
                            action = self._clear_aim_action(action)

                        last_metrics = None
                        if computed is not None:
                            aim_action, metrics = computed
                            action[self.action_layout.yaw_index] = aim_action[self.action_layout.yaw_index]
                            action[self.action_layout.pitch_index] = aim_action[self.action_layout.pitch_index]
                            last_metrics = metrics.as_dict()

                        should_fire = last_metrics is not None and self.fire_when_aligned and self._is_aligned(last_metrics)
                        if should_fire and self.action_layout.fire_index is not None:
                            fire_action = action.copy()
                            fire_action[self.action_layout.fire_index] = 1.0
                            fired = session.step(fire_action)
                            last_info = fired.info
                            frame_rgb = session.decode_frame(fired.observation, last_info)
                            last_info = self._resolve_tracking_info(last_info, frame_rgb)
                            fire_info = last_info.get("fire", {})
                            if isinstance(fire_info, dict):
                                print(f"[FIRE] {fire_info}")

                        if step_idx % FRAME_SKIP == 0:
                            display = self._build_display(frame_rgb, last_info, last_metrics)
                            cv2.imshow("target-lock", display)

                        key = cv2.waitKey(1)
                        self._handle_action_mutator_key(key)
                        if (key & 0xFF) == 27:
                            break

                        time.sleep(CONTROL_DT)
                finally:
                    cv2.destroyAllWindows()
        finally:
            self._close_detector()

        return {"last_info": last_info, "last_metrics": last_metrics}

    def _is_aligned(self, metrics: dict[str, float]) -> bool:
        if abs(metrics["azimuth_deg"]) > self.threshold.azimuth_deg:
            return False
        if abs(metrics["elevation_deg"]) > self.threshold.elevation_deg:
            return False
        if self.threshold.plane_x is not None and abs(metrics.get("plane_x", 0.0)) > self.threshold.plane_x:
            return False
        if self.threshold.plane_y is not None and abs(metrics.get("plane_y", 0.0)) > self.threshold.plane_y:
            return False
        return True

    def _clear_aim_action(self, action: np.ndarray) -> np.ndarray:
        action[self.action_layout.yaw_index] = 0.0
        action[self.action_layout.pitch_index] = 0.0
        if self.action_layout.fire_index is not None:
            action[self.action_layout.fire_index] = 0.0
        return action

    def _handle_action_mutator_key(self, key: int) -> None:
        mutator = self.action_mutator
        if mutator is None:
            return
        handle_key = getattr(mutator, "handle_key", None)
        if callable(handle_key):
            handle_key(key)

    def _resolve_tracking_info(
        self,
        info: dict[str, object],
        frame_rgb: np.ndarray,
    ) -> dict[str, object]:
        resolved = dict(info)
        bullseye_pixel = resolved.get("bullseye_pixel")
        if isinstance(bullseye_pixel, list) and len(bullseye_pixel) == 2:
            resolved["oracle_bullseye_pixel"] = [float(bullseye_pixel[0]), float(bullseye_pixel[1])]

        if self.bullseye_detector is None:
            resolved["bullseye_source"] = self.bullseye_source.value
            if self.bullseye_source == BullseyeSource.VISION:
                resolved.pop("bullseye_pixel", None)
                resolved.pop("vision_bullseye_score", None)
                resolved.pop("vision_bullseye_norm", None)
            return resolved

        if self.bullseye_source == BullseyeSource.VISION:
            detection = self._detect_vision_bullseye(frame_rgb, resolved)
            return self._apply_bullseye_detection(resolved, detection)

        detection = self._detect_bullseye(frame_rgb, resolved)
        return self._apply_oracle_detection(resolved, detection)

    def _detect_vision_bullseye(
        self,
        frame_rgb: np.ndarray,
        info: dict[str, object],
    ) -> BullseyeDetection | None:
        should_detect = self._vision_frame_index % self.vision_detect_every_n_frames == 0
        self._vision_frame_index += 1
        if should_detect:
            detection = self._detect_bullseye(frame_rgb, info)
            if self._uses_polled_detection():
                self._last_vision_detection = detection
            else:
                self._last_vision_detection = self._smooth_vision_detection(detection, info, frame_rgb.shape)
            return self._last_vision_detection
        has_latest_detection, detection = self._get_latest_bullseye_detection()
        if has_latest_detection:
            self._last_vision_detection = detection
        return self._last_vision_detection

    def _smooth_vision_detection(
        self,
        detection: BullseyeDetection | None,
        info: dict[str, object],
        frame_shape: tuple[int, int, int],
    ) -> BullseyeDetection | None:
        if detection is None:
            return None

        previous = self._last_vision_detection
        if previous is None or self.vision_smoothing_alpha >= 1.0:
            return detection

        alpha = self.vision_smoothing_alpha
        width = int(info.get("width", frame_shape[1]))
        height = int(info.get("height", frame_shape[0]))
        pixel_x = alpha * detection.pixel_x + (1.0 - alpha) * previous.pixel_x
        pixel_y = alpha * detection.pixel_y + (1.0 - alpha) * previous.pixel_y
        return build_detection(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            width=width,
            height=height,
            score=detection.score,
        )

    def _detect_bullseye(
        self,
        frame_rgb: np.ndarray,
        info: dict[str, object],
    ) -> BullseyeDetection | None:
        if self.bullseye_detector is None:
            return None

        detect = self.bullseye_detector.detect
        try:
            parameters = inspect.signature(detect).parameters
        except (TypeError, ValueError):
            return detect(frame_rgb)
        if "info" in parameters:
            return detect(frame_rgb, info=info)
        return detect(frame_rgb)

    def _get_latest_bullseye_detection(self) -> tuple[bool, BullseyeDetection | None]:
        detector = self.bullseye_detector
        if detector is None:
            return False, None
        get_latest_detection = getattr(detector, "get_latest_detection", None)
        if not callable(get_latest_detection):
            return False, None
        return True, get_latest_detection()

    def _uses_polled_detection(self) -> bool:
        has_latest_detection, _ = self._get_latest_bullseye_detection()
        return has_latest_detection

    def _reset_detector(self) -> None:
        detector = self.bullseye_detector
        if detector is None:
            return
        reset = getattr(detector, "reset", None)
        if callable(reset):
            reset()

    def _close_detector(self) -> None:
        detector = self.bullseye_detector
        if detector is None:
            return
        close = getattr(detector, "close", None)
        if callable(close):
            close()

    def _apply_oracle_detection(
        self,
        info: dict[str, object],
        detection: BullseyeDetection | None,
    ) -> dict[str, object]:
        resolved = dict(info)
        resolved["bullseye_source"] = BullseyeSource.ORACLE.value
        resolved.pop("vision_bullseye_score", None)
        resolved.pop("vision_bullseye_norm", None)
        if detection is None:
            return resolved
        resolved["bullseye_pixel"] = detection.to_pixel_list()
        return resolved

    def _apply_bullseye_detection(
        self,
        info: dict[str, object],
        detection: BullseyeDetection | None,
    ) -> dict[str, object]:
        resolved = dict(info)
        resolved["bullseye_source"] = BullseyeSource.VISION.value
        resolved.pop("bullseye_pixel", None)
        resolved.pop("vision_bullseye_score", None)
        resolved.pop("vision_bullseye_norm", None)
        if detection is None:
            return resolved

        resolved["bullseye_pixel"] = detection.to_pixel_list()
        resolved["vision_bullseye_score"] = detection.score
        resolved["vision_bullseye_norm"] = [detection.x_norm, detection.y_norm]
        return resolved

    def _build_display(
        self,
        frame_rgb: np.ndarray,
        info: dict[str, object],
        metrics: dict[str, float] | None,
    ) -> np.ndarray:
        import cv2

        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._draw_overlay(frame, info, metrics)
        schematic = self._render_schematic(info, frame.shape[0])
        return np.concatenate([frame, schematic], axis=1)

    def _draw_overlay(self, frame: np.ndarray, info: dict[str, object], metrics: dict[str, float] | None) -> None:
        import cv2

        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2
        cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

        bullseye_source = info.get("bullseye_source")
        bullseye_pixel = info.get("bullseye_pixel")
        if bullseye_source == BullseyeSource.VISION.value and isinstance(bullseye_pixel, list) and len(bullseye_pixel) == 2:
            cv2.circle(
                frame,
                (int(float(bullseye_pixel[0])), int(float(bullseye_pixel[1]))),
                4,
                (0, 255, 255),
                1,
            )

        oracle_bullseye_pixel = info.get("oracle_bullseye_pixel")
        if isinstance(oracle_bullseye_pixel, list) and len(oracle_bullseye_pixel) == 2:
            cv2.circle(
                frame,
                (int(float(oracle_bullseye_pixel[0])), int(float(oracle_bullseye_pixel[1]))),
                3,
                (255, 0, 0),
                -1,
            )

        lines = []
        if isinstance(bullseye_source, str):
            lines.append(f"target_src={bullseye_source}")
        vision_score = info.get("vision_bullseye_score")
        if isinstance(vision_score, (float, int)):
            lines.append(f"vision_score={float(vision_score):.3f}")
        if metrics is not None:
            lines.extend(f"{key}={value:.3f}" for key, value in metrics.items())
        qpos = info.get("qpos")
        if isinstance(qpos, list) and len(qpos) >= 5:
            lines.append(
                f"qpos=({float(qpos[0]):.3f}, {float(qpos[1]):.3f}, {float(qpos[2]):.3f}, {float(qpos[3]):.3f}, {float(qpos[4]):.3f})"
            )
        for idx, line in enumerate(lines[:9]):
            cv2.putText(frame, line, (12, 24 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _world_to_panel(
        self,
        x: float,
        y: float,
        panel_width: int,
        panel_height: int,
        padding: int = 24,
    ) -> tuple[int, int]:
        x0, x1 = SCHEMATIC_WORLD_X
        y0, y1 = SCHEMATIC_WORLD_Y
        px = padding + (x - x0) / (x1 - x0) * (panel_width - 2 * padding)
        py = panel_height - padding - (y - y0) / (y1 - y0) * (panel_height - 2 * padding)
        return int(px), int(py)

    def _render_schematic(self, info: dict[str, object], frame_height: int, panel_width: int = 320) -> np.ndarray:
        import cv2

        panel = np.full((frame_height, panel_width, 3), 248, dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (panel_width - 1, frame_height - 1), (210, 210, 210), 1)
        cv2.putText(panel, "Top View", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

        tx, ty = self._world_to_panel(SCHEMATIC_TARGET_X, SCHEMATIC_TARGET_Y, panel_width, frame_height)
        cv2.circle(panel, (tx, ty), 7, (30, 30, 220), -1)
        cv2.putText(panel, "target", (tx + 10, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 220), 1, cv2.LINE_AA)

        qpos = info.get("qpos", [])
        if isinstance(qpos, list) and len(qpos) >= 5:
            base_x = float(qpos[0])
            base_y = float(qpos[1])
            base_yaw = float(qpos[2])
            turret_yaw = float(qpos[3])
            pitch = float(qpos[4])
            facing_yaw = base_yaw + turret_yaw

            bx, by = self._world_to_panel(base_x, base_y, panel_width, frame_height)
            cv2.circle(panel, (bx, by), 6, (30, 160, 30), -1)
            cv2.putText(panel, "turret", (bx + 10, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 160, 30), 1, cv2.LINE_AA)

            dir_len = 0.28
            dx = dir_len * np.cos(facing_yaw)
            dy = dir_len * np.sin(facing_yaw)
            ex, ey = self._world_to_panel(base_x + dx, base_y + dy, panel_width, frame_height)
            cv2.line(panel, (bx, by), (ex, ey), (20, 120, 20), 2)
            cv2.circle(panel, (ex, ey), 4, (20, 120, 20), -1)
            cv2.line(panel, (bx, by), (tx, ty), (160, 160, 160), 1)

            cv2.putText(panel, f"base=({base_x:.2f}, {base_y:.2f})", (16, frame_height - 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(panel, f"base_yaw={base_yaw:.2f} gun_yaw={turret_yaw:.2f}", (16, frame_height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(panel, f"facing={facing_yaw:.2f} pitch={pitch:.2f}", (16, frame_height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

        return panel


__all__ = [
    "AlignmentThreshold",
    "BullseyeSource",
    "Runner",
]
