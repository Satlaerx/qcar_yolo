"""src/train/train.py
Fine-tune YOLOv11 on the traffic sign dataset.

Usage:
    python src/train/train.py
    python src/train/train.py --config configs/train.yaml
    python src/train/train.py --resume models/runs/traffic_sign_v1/weights/last.pt
"""

import argparse
from pathlib import Path
import yaml


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(cfg_path: str, resume: str | None = None):
    from ultralytics import YOLO

    cfg = load_cfg(cfg_path)

    model_weight = resume if resume else cfg.pop("model", "yolo11n.pt")
    # Check pretrained dir first
    pretrained_path = Path("models/pretrained") / model_weight
    if pretrained_path.exists():
        model_weight = str(pretrained_path)

    model = YOLO(model_weight)

    print(f"Model   : {model_weight}")
    print(f"Data    : {cfg['data']}")
    print(f"Epochs  : {cfg.get('epochs', 100)}")
    print(f"Batch   : {cfg.get('batch', 16)}")
    print(f"Device  : {cfg.get('device', 0)}")
    print()

    results = model.train(**cfg, resume=bool(resume))
    print("\nTraining complete.")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to last.pt to resume training")
    args = parser.parse_args()

    train(args.config, args.resume)