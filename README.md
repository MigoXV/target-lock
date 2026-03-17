# target-lock

`target-lock` is a target-locking controller demo package that integrates with the `lockon` simulator over gRPC.

## Features

- Back-projection utilities for converting image-plane observations into aiming geometry
- Open-loop aiming controllers for turret yaw and pitch
- PID-based tracking demo for moving platform scenarios
- CLI entrypoints for static, moving, and square-trajectory demos

## Install

```bash
pip install -e .
```

If `lockon` is not installed in the active environment, you can point to its source tree:

```powershell
$env:TARGET_LOCK_LOCKON_PATH="D:\academic\python\lockon\src"
```

## Run

Unified Typer CLI:

```bash
target-lock static --server-addr 127.0.0.1:50051
target-lock move --server-addr 127.0.0.1:50051
target-lock square-pid --server-addr 127.0.0.1:50051
```

Legacy entrypoints are still supported:

```bash
target-lock-static --server-addr 127.0.0.1:50051
target-lock-move --server-addr 127.0.0.1:50051
target-lock-square-pid --server-addr 127.0.0.1:50051
```

The `lockon` environment expects a 6-element action vector:

```text
[move_x, move_y, base_rot, turret_yaw, turret_pitch, fire]
```
