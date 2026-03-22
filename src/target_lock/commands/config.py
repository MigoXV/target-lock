from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(slots=True)
class ServerConfig:
    addr: str = "127.0.0.1:50051"


@dataclass(slots=True)
class SessionConfig:
    max_steps: int | None = None
    fire_when_aligned: bool = True


@dataclass(slots=True)
class AlignmentConfig:
    azimuth_deg: float = 0.18
    elevation_deg: float = 0.18
    plane_x: float | None = 0.01
    plane_y: float | None = 0.01


@dataclass(slots=True)
class TrackingConfig:
    bullseye_source: str = "vision"
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)


@dataclass(slots=True)
class OpenLoopConfig:
    yaw_step_rad: float = 0.08
    pitch_step_rad: float = 0.08


@dataclass(slots=True)
class AxisPidConfig:
    kp: float = 2.4
    ki: float = 0.05
    kd: float = 0.32


@dataclass(slots=True)
class PidConfig:
    yaw: AxisPidConfig = field(default_factory=AxisPidConfig)
    pitch: AxisPidConfig = field(default_factory=AxisPidConfig)
    deadband: float = 0.001
    integral_limit: float = 0.25
    feedback_limit: float = 0.65


@dataclass(slots=True)
class ControllerConfig:
    open_loop: OpenLoopConfig = field(default_factory=OpenLoopConfig)
    scan_yaw_command: float = 0.2
    scan_limit_deg: float = 85.0
    pid: PidConfig = field(default_factory=PidConfig)


@dataclass(slots=True)
class MotionConfig:
    manual_base_control: bool = False
    move_speed: float = 0.08
    base_rot_scale: float = 0.025
    hold_steps: int = 120
    seed: int = 0


@dataclass(slots=True)
class VisionConfig:
    onnx_path: str | None = None
    img_size_fallback: int = 640
    score_threshold: float = 0.0
    detect_every_n_frames: int = 1
    smoothing_alpha: float = 1.0
    async_inference: bool = True


@dataclass(slots=True)
class MoveCommandConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


def load_move_config(config_path: str | Path) -> MoveCommandConfig:
    structured = OmegaConf.structured(MoveCommandConfig)
    loaded = OmegaConf.load(Path(config_path))
    merged = OmegaConf.merge(structured, loaded)
    OmegaConf.resolve(merged)
    config = OmegaConf.to_object(merged)
    if not isinstance(config, MoveCommandConfig):
        raise TypeError(f"Expected MoveCommandConfig, got {type(config)!r}")
    return config
