"""src/inference/detect_image.py
Run YOLOv11 inference on a single image, folder, or video file.

Usage:
    python src/inference/detect_image.py --source data/splits/test/images
    python src/inference/detect_image.py --source path/to/image.jpg --show
    python src/inference/detect_image.py --source 0  # webcam
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def run_inference(weights: str, source: str, conf: float, iou: float,
                  show: bool, save: bool, save_dir: str):
    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        show=show,
        save=save,
        project=save_dir,
        name="detect",
        exist_ok=True,
        stream=True,        # memory-efficient for large batches
    )

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                conf_v = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                print(f"  [{label}] conf={conf_v:.2f}  bbox={[round(c,1) for c in coords]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  default="models/runs/traffic_sign_v1/weights/best.pt")
    parser.add_argument("--source",   default="data/splits/test/images")
    parser.add_argument("--conf",     type=float, default=0.25)
    parser.add_argument("--iou",      type=float, default=0.45)
    parser.add_argument("--show",     action="store_true", help="Display results")
    parser.add_argument("--save",     action="store_true", help="Save output images")
    parser.add_argument("--save_dir", default="models/runs")
    args = parser.parse_args()

    run_inference(**vars(args))
