from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


DEFAULT_AUTOAIM_MODEL = Path("yolo") / "point_yolo_v8.onnx"
LEGACY_AUTOAIM_MODEL = Path("point_yolo.onnx")
AUTOAIM_REPO_ENV_VARS = ("TARGET_LOCK_AUTOAIM_REPO", "AUTOAIM_REPO")
AUTOAIM_ONNX_ENV_VARS = ("TARGET_LOCK_ONNX_PATH", "ONNX_PATH")


@dataclass(frozen=True, slots=True)
class BullseyeDetection:
    pixel_x: float
    pixel_y: float
    score: float
    x_norm: float
    y_norm: float

    def to_pixel_list(self) -> list[float]:
        return [self.pixel_x, self.pixel_y]


class BullseyeDetector(Protocol):
    def detect(self, frame_rgb: np.ndarray) -> BullseyeDetection | None:
        ...


def _find_project_dotenv() -> Path | None:
    cwd = Path.cwd().resolve()
    for directory in (cwd, *cwd.parents):
        dotenv_path = directory / ".env"
        if dotenv_path.is_file():
            return dotenv_path
    return None


def _read_dotenv_values() -> dict[str, str]:
    dotenv_path = _find_project_dotenv()
    if dotenv_path is None:
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value

    return values


def _get_env_path(env_names: tuple[str, ...]) -> str | None:
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value

    dotenv_values = _read_dotenv_values()
    for env_name in env_names:
        value = dotenv_values.get(env_name)
        if value:
            return value
    return None


def resolve_autoaim_repo(autoaim_repo: str | Path | None = None) -> Path:
    if autoaim_repo is not None:
        return Path(autoaim_repo).expanduser()

    configured_repo = _get_env_path(AUTOAIM_REPO_ENV_VARS)
    if configured_repo is not None:
        return Path(configured_repo).expanduser()

    env_hint = ", ".join(AUTOAIM_REPO_ENV_VARS)
    onnx_hint = ", ".join(AUTOAIM_ONNX_ENV_VARS)
    raise ValueError(
        "Autoaim model location is not configured. "
        f"Pass --autoaim-repo or --onnx-path, or set one of: {env_hint}, {onnx_hint}."
    )


def resolve_autoaim_onnx_path(autoaim_repo: str | Path | None, onnx_path: str | Path | None = None) -> Path:
    if onnx_path is None:
        configured_onnx_path = _get_env_path(AUTOAIM_ONNX_ENV_VARS)
        if configured_onnx_path is not None:
            onnx_path = configured_onnx_path

    if onnx_path is not None:
        resolved = Path(onnx_path).expanduser()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"YOLO onnx model not found: {resolved}")

    repo_path = resolve_autoaim_repo(autoaim_repo)
    candidates = [
        repo_path / DEFAULT_AUTOAIM_MODEL,
        repo_path / LEGACY_AUTOAIM_MODEL,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Unable to find autoaim YOLO model. Tried: {searched}")


def _letterbox_frame(
    image_rgb: np.ndarray,
    size: int,
    pad_value: int = 114,
) -> tuple[np.ndarray, dict[str, float | int]]:
    height, width = image_rgb.shape[:2]
    scale = min(size / height, size / width)
    resized_height = int(round(height * scale))
    resized_width = int(round(width * scale))
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_w = (size - resized_width) // 2
    pad_h = (size - resized_height) // 2
    canvas[pad_h:pad_h + resized_height, pad_w:pad_w + resized_width] = resized
    return canvas, {
        "scale": scale,
        "pad_w": pad_w,
        "pad_h": pad_h,
        "orig_w": width,
        "orig_h": height,
    }


def _preprocess_frame(frame_rgb: np.ndarray, img_size: int) -> tuple[np.ndarray, dict[str, float | int]]:
    resized, meta = _letterbox_frame(frame_rgb, img_size)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, meta


def _postprocess_point(
    point_xy: np.ndarray,
    meta: dict[str, float | int],
    img_size: int,
) -> tuple[float, float, float, float]:
    x_model = float(point_xy[0]) * img_size
    y_model = float(point_xy[1]) * img_size

    x_orig = (x_model - float(meta["pad_w"])) / float(meta["scale"])
    y_orig = (y_model - float(meta["pad_h"])) / float(meta["scale"])

    x_orig = float(np.clip(x_orig, 0.0, float(meta["orig_w"]) - 1.0))
    y_orig = float(np.clip(y_orig, 0.0, float(meta["orig_h"]) - 1.0))

    x_norm = x_orig / float(meta["orig_w"])
    y_norm = y_orig / float(meta["orig_h"])
    return x_orig, y_orig, x_norm, y_norm


def _create_session(onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "Vision target detection requires `onnxruntime`. Install it with `pip install onnxruntime`."
        ) from exc

    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(onnx_path, providers=providers)


def _resolve_img_size(session, fallback: int) -> int:
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) >= 4:
        height = input_shape[2]
        width = input_shape[3]
        if isinstance(height, int) and isinstance(width, int) and height == width:
            return int(height)
    return fallback


class YoloBullseyeDetector:
    def __init__(
        self,
        *,
        autoaim_repo: str | Path | None = None,
        onnx_path: str | Path | None = None,
        img_size_fallback: int = 640,
        score_threshold: float = 0.0,
    ) -> None:
        self.onnx_path = resolve_autoaim_onnx_path(autoaim_repo, onnx_path)
        self.score_threshold = float(score_threshold)
        self.session = _create_session(str(self.onnx_path))
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = _resolve_img_size(self.session, img_size_fallback)

    def detect(self, frame_rgb: np.ndarray) -> BullseyeDetection | None:
        tensor, meta = _preprocess_frame(frame_rgb, self.img_size)
        points, scores = self.session.run(None, {self.input_name: tensor})

        point_xy = points[0]
        score = float(scores[0, 0])
        if score < self.score_threshold:
            return None

        x_orig, y_orig, x_norm, y_norm = _postprocess_point(point_xy, meta, self.img_size)
        return BullseyeDetection(
            pixel_x=x_orig,
            pixel_y=y_orig,
            score=score,
            x_norm=x_norm,
            y_norm=y_norm,
        )
