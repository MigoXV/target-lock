from __future__ import annotations

import os
import queue
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


DEFAULT_SERVER_ADDR = os.getenv("LOCKON_SERVER_ADDR", "127.0.0.1:50051")
STREAM_END = object()


@dataclass(frozen=True, slots=True)
class LockonPaths:
    repo_root: Path
    src_root: Path


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Any
    info: dict[str, object]
    reward: float
    terminated: bool
    truncated: bool


def ensure_lockon_importable() -> LockonPaths:
    env_path = os.getenv("TARGET_LOCK_LOCKON_PATH")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd.parent / "lockon" / "src",
            cwd.parent / "lockon",
            Path(r"D:\academic\python\lockon\src"),
            Path(r"D:\academic\python\lockon"),
        ]
    )

    for candidate in candidates:
        src_root = candidate if candidate.name == "src" else candidate / "src"
        if (src_root / "lockon").exists():
            resolved = str(src_root.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return LockonPaths(repo_root=src_root.parent, src_root=src_root)

    raise ModuleNotFoundError(
        "Unable to locate the lockon package. Set TARGET_LOCK_LOCKON_PATH to the lockon src directory."
    )


class LockonSession:
    def __init__(self, server_addr: str = DEFAULT_SERVER_ADDR) -> None:
        ensure_lockon_importable()
        import grpc
        from google.protobuf.json_format import MessageToDict
        from lockon.protos.gym_env import gym_env_pb2, gym_env_pb2_grpc
        from lockon.utils import array_from_tensor, tensor_from_array
        from lockon.vcodec import create_observation_decoder

        self.grpc = grpc
        self.MessageToDict = MessageToDict
        self.gym_env_pb2 = gym_env_pb2
        self.gym_env_pb2_grpc = gym_env_pb2_grpc
        self.array_from_tensor = array_from_tensor
        self.tensor_from_array = tensor_from_array
        self.create_observation_decoder = create_observation_decoder
        self.server_addr = server_addr
        self.request_queue: queue.Queue[object] = queue.Queue()
        self.channel = None
        self.responses = None
        self.decoder = None

    def _request_iterator(self) -> Iterator[Any]:
        while True:
            item = self.request_queue.get()
            if item is STREAM_END:
                return
            yield item

    def __enter__(self) -> "LockonSession":
        self.channel = self.grpc.insecure_channel(self.server_addr)
        self.stub = self.gym_env_pb2_grpc.ArmEnvStub(self.channel)
        self.responses = self.stub.StreamEnv(self._request_iterator())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.responses is not None:
                self.request_queue.put(self.gym_env_pb2.EnvRequest(close=self.gym_env_pb2.Close()))
                try:
                    next(self.responses)
                except StopIteration:
                    pass
        finally:
            self.request_queue.put(STREAM_END)
            if self.decoder is not None:
                self.decoder.close()
                self.decoder = None
            if self.channel is not None:
                self.channel.close()
                self.channel = None

    def reset(self) -> np.ndarray:
        self.request_queue.put(self.gym_env_pb2.EnvRequest(reset=self.gym_env_pb2.Reset()))
        reply = next(self.responses)
        if reply.WhichOneof("result") != "reset":
            raise RuntimeError("expected ResetReply")
        return self.decode_frame(reply.reset.observation, {})

    def decode_frame(self, observation: Any, info: dict[str, object]) -> np.ndarray:
        if self.decoder is None or getattr(self.decoder, "tensor_dtype", None) != observation.dtype:
            if self.decoder is not None:
                self.decoder.close()
            self.decoder = self.create_observation_decoder(observation.dtype)
            self.decoder.reset()
        return self.decoder.decode(observation, info)

    def step(self, action: np.ndarray) -> StepResult:
        self.request_queue.put(
            self.gym_env_pb2.EnvRequest(
                step=self.gym_env_pb2.Step(action=self.tensor_from_array(action.astype(np.float32)))
            )
        )
        reply = next(self.responses)
        if reply.WhichOneof("result") != "step":
            raise RuntimeError("expected StepReply")
        info = self.MessageToDict(reply.step.info, preserving_proto_field_name=True)
        return StepResult(
            observation=reply.step.observation,
            info=info,
            reward=float(self.array_from_tensor(reply.step.reward)),
            terminated=bool(self.array_from_tensor(reply.step.terminated)),
            truncated=bool(self.array_from_tensor(reply.step.truncated)),
        )
