# Traffic Sign Detection & Recognition — YOLOv11 + QLabs QCar2

基于 YOLOv11 的交通标志检测与识别系统，集成 Quanser QLabs QCar2 仿真平台，实现自动驾驶数据采集、模型训练和实时推理。

## 项目结构

```
traffic_sign_yolo/
├── configs/
│   ├── dataset.yaml        # 数据集路径与类别定义
│   └── train.yaml          # 训练超参数
├── data/
│   ├── raw/                # QLabs 采集的原始图像
│   ├── annotated/          # LabelImg 标注后的数据（YOLO格式）
│   └── splits/             # train / val / test 分割结果
├── models/
│   ├── pretrained/         # YOLOv11 预训练权重
│   └── runs/               # 训练输出（权重、日志、可视化）
├── src/
│   ├── data/
│   │   └── split_dataset.py        # 数据集分割工具
│   ├── qlab/
│   │   ├── setup_scene.py          # QLabs 场景初始化 + 交通标志布置
│   │   ├── collect_keyboard.py     # 键盘手动控制采集
│   │   └── collect_auto.py         # 自动驾驶采集（QCarDriveController）
│   ├── train/
│   │   └── train.py                # YOLOv11 微调训练
│   └── inference/
│       └── detect_image.py         # 图像/视频推理
├── scripts/
│   ├── setup_env.bat               # Windows 一键环境配置
│   └── download_weights.py         # 下载 YOLOv11 预训练权重
├── docs/
│   └── class_list.md               # 交通标志类别定义
└── requirements.txt
```

## 环境要求

- Windows 10/11
- Python 3.10 或 3.11（**系统 Python，不使用 venv**）
- Quanser QLabs 已安装（提供 `qvl`、`pal`、`hal` 库）
- CUDA GPU（训练推荐，推理可用 CPU）

> ⚠️ Quanser 库为闭源私有库，随 QLabs 软件安装，无法通过 pip 安装。
> 所有 QLabs 相关脚本须使用系统 Python 运行，不要使用 venv。

## 快速开始

### 1. 安装依赖

```bat
D:\Python\python.exe -m pip install ultralytics opencv-python keyboard
```

### 2. 下载 YOLOv11 预训练权重

```bat
D:\Python\python.exe scripts\download_weights.py --model yolo11n.pt
```

### 3. 布置 QLabs 场景

打开 QLabs 并加载 Cityscape 场景，然后运行：

```bat
D:\Python\python.exe src\qlab\setup_scene.py
```

场景中会自动生成停车标志、让行标志、环岛标志等。运行完后在 QLabs 中 `File → Save Scene` 保存布局。

### 4. 采集数据

**自动驾驶采集（推荐）**：QCar2 沿 QLabs 内置路网自动行驶，图像保存至 `data/raw/`：

```bat
D:\Python\python.exe src\qlab\collect_auto.py --loops 5 --no-setup
```

**手动键盘采集**：用键盘控制 QCar2 行驶（需管理员权限运行终端）：

```bat
D:\Python\python.exe src\qlab\collect_keyboard.py --no-setup
```

键盘说明：W/S/A/D 控制方向，空格停车，C 手动存图，Q 退出。

### 5. 标注图像

安装并运行 LabelImg：

```bat
D:\Python\python.exe -m pip install labelImg
D:\Python\python.exe -m labelImg
```

- Open Dir → `data\raw\`
- Change Save Dir → `data\annotated\labels\`
- 格式选 **YOLO**
- 快捷键：**W** 画框，**D** 下一张

### 6. 分割数据集

```bat
D:\Python\python.exe src\data\split_dataset.py
```

按 7:2:1 自动分割到 `data/splits/`。

### 7. 训练

```bat
D:\Python\python.exe src\train\train.py
```

### 8. 推理测试

```bat
D:\Python\python.exe src\inference\detect_image.py --source data\splits\test\images --show
```

## 交通标志类别

详见 `docs/class_list.md`，默认包含 10 类：

| ID | 类别 | ID | 类别 |
|----|------|----|------|
| 0  | stop（停车）| 5 | speed_limit_80 |
| 1  | yield（让行）| 6 | no_entry（禁止进入）|
| 2  | speed_limit_30 | 7 | turn_left |
| 3  | speed_limit_50 | 8 | turn_right |
| 4  | speed_limit_60 | 9 | pedestrian_crossing |

## 控制架构说明

数据采集使用 Quanser 官方控制框架：

```
SDCSRoadMap           → 读取内置路网，生成路径点（不会撞墙）
QCarDriveController   → 速度 PI + Stanley 转向一体控制
QCarEKF + QCarGPS     → 实时位置估计
Camera2D              → CSI 摄像头图像读取
```