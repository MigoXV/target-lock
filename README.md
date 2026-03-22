# target-lock

`target-lock` 是一个目标锁定控制器演示包，通过 Gym V2 gRPC 接口与 `lockon` 模拟器集成。

## 功能

- 提供将图像平面观测转换为瞄准几何量的反投影工具
- 提供面向运动平台场景的 PID 跟踪演示
- 提供基于 OmegaConf 的 `move` 配置，并附带默认示例 YAML

## 安装

```bash
pip install -e .
```

项目已内置所需的 Gym V2 protobuf 绑定，因此不需要把本地 `lockon` 源码目录加入 `PYTHONPATH`。

## 运行

统一的 Typer CLI：

```bash
target-lock move
target-lock move --config examples/move/config.yaml
```

也可以通过环境变量开启手动底座控制：

```bash
export TARGET_LOCK_MANUAL_BASE_CONTROL=1
target-lock move
```

手动模式下使用 `W/S` 控制前后、`A/D` 控制左右、`Q/E` 控制底座旋转，空格停止底座动作。

仍然兼容旧入口：

```bash
target-lock-move
```

默认配置文件位于 `examples/move/config.yaml`。
建议在这里调整运动、PID、对齐阈值、视觉和运行时参数，而不是在命令行里传很长的参数列表。

视觉配置里的 `vision.async_inference` 默认为 `true`。
开启后 YOLO/ONNX 推理会在后台线程里执行，主控制循环不会阻塞等待检测结果。
该模式采用 latest-only 策略，只保留最新待处理帧，不保证逐帧检测；`vision.detect_every_n_frames` 仍然生效，控制器会复用最近一次已完成的视觉结果。

```powershell
$env:TARGET_LOCK_ONNX_PATH="D:\academic\python\autoaim\yolo\point_yolo_v8.onnx"
```

示例配置默认通过 OmegaConf 插值读取 `TARGET_LOCK_ONNX_PATH`。

`lockon` 环境要求动作为 6 维向量：

```text
[move_x, move_y, base_rot, turret_yaw, turret_pitch, fire]
```

远端 gRPC 服务需要暴露 `gym_v2.GymEnv/StreamEnv`。
