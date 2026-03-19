from __future__ import annotations

from pathlib import Path

import pytest

from target_lock.vision import DEFAULT_AUTOAIM_MODEL, resolve_autoaim_onnx_path


def _create_model_file(repo_dir: Path) -> Path:
    model_path = repo_dir / DEFAULT_AUTOAIM_MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"onnx")
    return model_path


def test_resolve_autoaim_onnx_path_prefers_explicit_path(tmp_path: Path) -> None:
    explicit_model = tmp_path / "custom" / "model.onnx"
    explicit_model.parent.mkdir(parents=True, exist_ok=True)
    explicit_model.write_bytes(b"onnx")

    resolved = resolve_autoaim_onnx_path(None, explicit_model)

    assert resolved == explicit_model


def test_resolve_autoaim_onnx_path_requires_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TARGET_LOCK_AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("TARGET_LOCK_ONNX_PATH", raising=False)
    monkeypatch.delenv("ONNX_PATH", raising=False)

    with pytest.raises(ValueError, match="Autoaim model location is not configured"):
        resolve_autoaim_onnx_path(None, None)


def test_resolve_autoaim_onnx_path_reads_repo_from_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "env-autoaim"
    expected_model = _create_model_file(repo_dir)

    monkeypatch.setenv("TARGET_LOCK_AUTOAIM_REPO", str(repo_dir))
    monkeypatch.delenv("AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("TARGET_LOCK_ONNX_PATH", raising=False)
    monkeypatch.delenv("ONNX_PATH", raising=False)

    resolved = resolve_autoaim_onnx_path(None, None)

    assert resolved == expected_model


def test_resolve_autoaim_onnx_path_reads_repo_from_dotenv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "dotenv-autoaim"
    expected_model = _create_model_file(repo_dir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TARGET_LOCK_AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("TARGET_LOCK_ONNX_PATH", raising=False)
    monkeypatch.delenv("ONNX_PATH", raising=False)
    (tmp_path / ".env").write_text(f"TARGET_LOCK_AUTOAIM_REPO={repo_dir}\n", encoding="utf-8")

    resolved = resolve_autoaim_onnx_path(None, None)

    assert resolved == expected_model


def test_resolve_autoaim_onnx_path_prefers_environment_over_dotenv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    env_repo_dir = tmp_path / "env-autoaim"
    dotenv_repo_dir = tmp_path / "dotenv-autoaim"
    expected_model = _create_model_file(env_repo_dir)
    _create_model_file(dotenv_repo_dir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARGET_LOCK_AUTOAIM_REPO", str(env_repo_dir))
    monkeypatch.delenv("AUTOAIM_REPO", raising=False)
    monkeypatch.delenv("TARGET_LOCK_ONNX_PATH", raising=False)
    monkeypatch.delenv("ONNX_PATH", raising=False)
    (tmp_path / ".env").write_text(f"TARGET_LOCK_AUTOAIM_REPO={dotenv_repo_dir}\n", encoding="utf-8")

    resolved = resolve_autoaim_onnx_path(None, None)

    assert resolved == expected_model
