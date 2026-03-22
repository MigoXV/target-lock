from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from target_lock.commands.config import load_move_config
from target_lock.runner.move import run_move


app = typer.Typer(
    help="Command-line entrypoints for target locking demos.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

DEFAULT_MOVE_CONFIG_PATH = Path("examples/move/config.yaml")

ConfigPathOption = Annotated[
    Path,
    typer.Option(
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the move command OmegaConf YAML.",
    ),
]

MaxStepsOption = Annotated[
    int | None,
    typer.Option(
        "--max-steps",
        min=1,
        help="Override the configured session step limit.",
    ),
]

FireOption = Annotated[
    bool | None,
    typer.Option(
        "--fire/--no-fire",
        help="Override whether the session fires automatically when aligned.",
    ),
]

ManualBaseControlOption = Annotated[
    bool | None,
    typer.Option(
        "--manual-base-control/--no-manual-base-control",
        envvar="TARGET_LOCK_MANUAL_BASE_CONTROL",
        help="Use WASD/QE keyboard input for base movement and rotation instead of random motion.",
    ),
]


@app.command("move")
def move_pid(
    config: ConfigPathOption = DEFAULT_MOVE_CONFIG_PATH,
    max_steps: MaxStepsOption = None,
    fire: FireOption = None,
    manual_base_control: ManualBaseControlOption = None,
) -> None:
    loaded = load_move_config(config)
    if max_steps is not None:
        loaded.session.max_steps = max_steps
    if fire is not None:
        loaded.session.fire_when_aligned = fire
    if manual_base_control is not None:
        loaded.motion.manual_base_control = manual_base_control
    run_move(loaded)


def legacy_move_main() -> None:
    app(prog_name="target-lock-move", args=["move"])


def main() -> None:
    app()


if __name__ == "__main__":
    main()
