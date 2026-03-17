from __future__ import annotations

from target_lock.commands.app import app


def main() -> None:
    app(prog_name="target-lock-move", args=["move"])


if __name__ == "__main__":
    main()
