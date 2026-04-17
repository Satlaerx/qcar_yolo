# Traffic Sign Detection & Recognition — YOLOv11 + QLabs

A complete pipeline for traffic sign detection and recognition using YOLOv11,
integrated with the Quanser QLabs simulation environment (QCar platform).

## Project Structure

```
traffic_sign_yolo/
├── configs/                  # YAML config files
│   ├── dataset.yaml          # Dataset paths and class definitions
│   └── train.yaml            # Training hyperparameters
├── data/
│   ├── raw/                  # Raw collected images / QLabs screenshots
│   ├── annotated/            # LabelImg / Label Studio export (YOLO format)
│   └── splits/               # train / val / test after split
│       ├── train/images|labels
│       ├── val/images|labels
│       └── test/images|labels
├── models/
│   ├── pretrained/           # Downloaded YOLOv11 weights (.pt)
│   └── runs/                 # Training experiment outputs
├── src/
│   ├── data/
│   │   ├── collect_qlab.py   # Capture frames from QLabs QCar camera
│   │   ├── split_dataset.py  # Train/val/test split utility
│   │   └── augment.py        # Optional offline augmentation
│   ├── train/
│   │   └── train.py          # Fine-tune YOLOv11
│   ├── inference/
│   │   ├── detect_image.py   # Single image / batch inference
│   │   └── detect_video.py   # Webcam / video stream inference
│   └── qlab/
│       └── qcar_inference.py # Real-time inference loop on QCar
├── notebooks/
│   └── explore.ipynb         # EDA, visualise annotations, evaluation
├── scripts/
│   ├── setup_env.bat         # Windows one-click env setup
│   └── download_weights.py   # Auto-download YOLOv11n/s weights
├── docs/
│   └── class_list.md         # Traffic sign class definitions
└── requirements.txt
```

## Quick Start

### 1. Setup environment (Windows)

```bat
scripts\setup_env.bat
```

Or manually:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download pretrained weights

```bat
python scripts\download_weights.py
```

### 3. Collect data from QLabs

```bat
python src\data\collect_qlab.py --output data\raw --num_frames 500
```

### 4. Annotate

Use [LabelImg](https://github.com/HumanSignal/labelImg) in YOLO format.
Export labels to `data/annotated/`.

### 5. Split dataset

```bat
python src\data\split_dataset.py --src data\annotated --dst data\splits
```

### 6. Train

```bat
python src\train\train.py --config configs\train.yaml
```

### 7. Real-time inference on QCar

```bat
python src\qlab\qcar_inference.py --weights models\runs\best.pt
```

## Classes

See `docs/class_list.md` for the full class list.
Default classes match Chinese/common international traffic signs.

## Requirements

- Python 3.10+
- Windows 10/11
- QLabs + Quanser QCar libraries (for simulation integration)
- CUDA-compatible GPU recommended for training
