"""scripts/download_weights.py
Download YOLOv11 pretrained weights from Ultralytics.
Saves to models/pretrained/.
"""

import sys
from pathlib import Path

WEIGHTS_DIR = Path("models/pretrained")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "yolo11n.pt": "nano  — fastest, good for edge/real-time",
    "yolo11s.pt": "small — balanced speed/accuracy",
    "yolo11m.pt": "medium",
}

def download_weights(model_name: str = "yolo11n.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    out_path = WEIGHTS_DIR / model_name
    if out_path.exists():
        print(f"[skip] {model_name} already exists at {out_path}")
        return

    print(f"Downloading {model_name} ...")
    # Ultralytics auto-downloads on first YOLO() call if not found locally
    model = YOLO(model_name)
    # Move cached weight to our pretrained dir
    import shutil, os
    cached = Path(model_name)
    if cached.exists():
        shutil.move(str(cached), str(out_path))
        print(f"Saved to {out_path}")
    else:
        print(f"Weight should be at: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt",
                        choices=list(MODELS.keys()),
                        help="Which YOLOv11 variant to download")
    args = parser.parse_args()

    print("Available variants:")
    for k, v in MODELS.items():
        marker = ">>>" if k == args.model else "   "
        print(f"  {marker} {k}: {v}")
    print()
    download_weights(args.model)
