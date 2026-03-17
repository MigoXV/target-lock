# target-lock

`target-lock` 是一个基于反投影算法的开环瞄准控制库，并提供了和 lockon 仿真工程对接的运行入口。

## 能力

- 反投影：把图像平面目标点转换成光轴坐标系方向向量与球坐标角
- 开环控制：按相机视场角和单步关节步长直接生成炮塔 yaw / pitch 控制量
- PID 混合控制：在开环前馈基础上叠加像平面误差反馈
- 仿真接入：复用 `lockon` 的 gRPC 环境流与视频解码
- CLI：提供静态目标与运动底盘两个示例入口

## 安装

```bash
pip install -e .[sim,dev]
```

如果 `lockon` 没有安装到当前环境，可以设置：

```bash
$env:TARGET_LOCK_LOCKON_PATH="D:\academic\python\lockon\src"
```

## 运行

静态场景：

```bash
target-lock-static --server-addr 127.0.0.1:50051
```

运动场景：

```bash
target-lock-move --server-addr 127.0.0.1:50051
```

???`lockon` ?? gRPC ????? `step.action` ??? 6 ??`target-lock` ??????????????????`[move_x, move_y, base_rot, turret_yaw, turret_pitch, fire]`

???? + PID ???

```bash
target-lock-square-pid --server-addr 127.0.0.1:50051
```
