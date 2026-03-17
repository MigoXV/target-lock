from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ActionLayout:
    size: int
    yaw_index: int
    pitch_index: int
    fire_index: int | None = None

    def build_idle(self) -> np.ndarray:
        return np.zeros(self.size, dtype=np.float32)


class AimMetrics(ABC):
    @abstractmethod
    def as_dict(self) -> dict[str, float]:
        raise NotImplementedError


class AimController(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        info: dict[str, Any],
        frame_shape: tuple[int, int, int],
        dt: float | None = None,
    ) -> tuple[np.ndarray, AimMetrics] | None:
        raise NotImplementedError
