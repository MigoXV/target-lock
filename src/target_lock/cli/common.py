from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from target_lock.controllers import ActionLayout, AimController
from target_lock.sim import LockonSession


CONTROL_DT = 0.01
FRAME_SKIP = 5
SCHEMATIC_TARGET_X = 0.5
SCHEMATIC_TARGET_Y = 0.0
SCHEMATIC_WORLD_X = (-1.0, 0.7)
SCHEMATIC_WORLD_Y = (-1.0, 1.0)


@dataclass(frozen=True, slots=True)
class AlignmentThreshold:
    azimuth_deg: float
    elevation_deg: float
    plane_x: float | None = None
    plane_y: float | None = None


def _is_aligned(metrics: dict[str, float], threshold: AlignmentThreshold) -> bool:
    if abs(metrics["azimuth_deg"]) > threshold.azimuth_deg:
        return False
    if abs(metrics["elevation_deg"]) > threshold.elevation_deg:
        return False
    if threshold.plane_x is not None and abs(metrics.get("plane_x", 0.0)) > threshold.plane_x:
        return False
    if threshold.plane_y is not None and abs(metrics.get("plane_y", 0.0)) > threshold.plane_y:
        return False
    return True


def _clear_aim_action(action: np.ndarray, action_layout: ActionLayout) -> np.ndarray:
    action[action_layout.yaw_index] = 0.0
    action[action_layout.pitch_index] = 0.0
    if action_layout.fire_index is not None:
        action[action_layout.fire_index] = 0.0
    return action


def run_session(
    *,
    server_addr: str,
    controller: AimController,
    action_layout: ActionLayout,
    max_steps: int | None,
    threshold: AlignmentThreshold,
    fire_when_aligned: bool,
    action_mutator: Callable[[int, np.ndarray], np.ndarray] | None = None,
) -> dict[str, object]:
    import cv2

    last_info: dict[str, object] = {}
    last_metrics: dict[str, float] | None = None
    aligned_last_step = False

    with LockonSession(server_addr=server_addr) as session:
        frame_rgb = session.reset()
        cv2.namedWindow("target-lock", cv2.WINDOW_NORMAL)
        action = action_layout.build_idle()
        step_indices = range(max_steps) if max_steps is not None else itertools.count()

        try:
            for step_idx in step_indices:
                if action_mutator is not None:
                    action = action_mutator(step_idx, action.copy())

                step_result = session.step(action)
                last_info = step_result.info
                frame_rgb = session.decode_frame(step_result.observation, last_info)

                computed = controller.update(last_info, frame_rgb.shape, dt=CONTROL_DT)
                if action_mutator is None:
                    action = action_layout.build_idle()
                else:
                    action = _clear_aim_action(action, action_layout)
                last_metrics = None
                if computed is not None:
                    aim_action, metrics = computed
                    action[action_layout.yaw_index] = aim_action[action_layout.yaw_index]
                    action[action_layout.pitch_index] = aim_action[action_layout.pitch_index]
                    last_metrics = metrics.as_dict()

                should_fire = False
                if last_metrics is not None:
                    aligned = _is_aligned(last_metrics, threshold)
                    should_fire = fire_when_aligned and aligned and not aligned_last_step
                    aligned_last_step = aligned
                else:
                    aligned_last_step = False

                if should_fire and action_layout.fire_index is not None:
                    fire_action = action.copy()
                    fire_action[action_layout.fire_index] = 1.0
                    fired = session.step(fire_action)
                    last_info = fired.info
                    frame_rgb = session.decode_frame(fired.observation, last_info)
                    fire_info = last_info.get("fire", {})
                    if isinstance(fire_info, dict):
                        print(f"[FIRE] {fire_info}")

                if step_idx % FRAME_SKIP == 0:
                    display = _build_display(frame_rgb, last_info, last_metrics)
                    cv2.imshow("target-lock", display)
                    key = cv2.waitKey(1)
                    if (key & 0xFF) == 27:
                        break

                time.sleep(CONTROL_DT)
        finally:
            cv2.destroyAllWindows()

    return {"last_info": last_info, "last_metrics": last_metrics}


def _build_display(frame_rgb: np.ndarray, info: dict[str, object], metrics: dict[str, float] | None) -> np.ndarray:
    import cv2

    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _draw_overlay(frame, info, metrics)
    schematic = _render_schematic(info, frame.shape[0])
    return np.concatenate([frame, schematic], axis=1)


def _draw_overlay(frame: np.ndarray, info: dict[str, object], metrics: dict[str, float] | None) -> None:
    import cv2

    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

    bullseye_pixel = info.get("bullseye_pixel")
    if isinstance(bullseye_pixel, list) and len(bullseye_pixel) == 2:
        cv2.circle(frame, (int(float(bullseye_pixel[0])), int(float(bullseye_pixel[1]))), 5, (255, 0, 0), -1)

    if metrics is None:
        return

    lines = [f"{key}={value:.3f}" for key, value in metrics.items()]
    qpos = info.get("qpos")
    if isinstance(qpos, list) and len(qpos) >= 5:
        lines.append(
            f"qpos=({float(qpos[0]):.3f}, {float(qpos[1]):.3f}, {float(qpos[2]):.3f}, {float(qpos[3]):.3f}, {float(qpos[4]):.3f})"
        )
    for idx, line in enumerate(lines[:9]):
        cv2.putText(frame, line, (12, 24 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _world_to_panel(
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


def _render_schematic(info: dict[str, object], frame_height: int, panel_width: int = 320) -> np.ndarray:
    import cv2

    panel = np.full((frame_height, panel_width, 3), 248, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, frame_height - 1), (210, 210, 210), 1)
    cv2.putText(panel, "Top View", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    tx, ty = _world_to_panel(SCHEMATIC_TARGET_X, SCHEMATIC_TARGET_Y, panel_width, frame_height)
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

        bx, by = _world_to_panel(base_x, base_y, panel_width, frame_height)
        cv2.circle(panel, (bx, by), 6, (30, 160, 30), -1)
        cv2.putText(panel, "turret", (bx + 10, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 160, 30), 1, cv2.LINE_AA)

        dir_len = 0.28
        dx = dir_len * np.cos(facing_yaw)
        dy = dir_len * np.sin(facing_yaw)
        ex, ey = _world_to_panel(base_x + dx, base_y + dy, panel_width, frame_height)
        cv2.line(panel, (bx, by), (ex, ey), (20, 120, 20), 2)
        cv2.circle(panel, (ex, ey), 4, (20, 120, 20), -1)
        cv2.line(panel, (bx, by), (tx, ty), (160, 160, 160), 1)

        cv2.putText(panel, f"base=({base_x:.2f}, {base_y:.2f})", (16, frame_height - 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(panel, f"base_yaw={base_yaw:.2f} gun_yaw={turret_yaw:.2f}", (16, frame_height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(panel, f"facing={facing_yaw:.2f} pitch={pitch:.2f}", (16, frame_height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

    return panel
