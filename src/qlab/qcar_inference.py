"""src/qlab/qcar_inference.py
Real-time traffic sign detection on the QLabs QCar.

Reads frames from the QCar front camera, runs YOLOv11 inference,
and prints / visualises detections. Optionally publishes results
back to QLabs for display overlay.

Usage:
    python src/qlab/qcar_inference.py --weights models/runs/traffic_sign_v1/weights/best.pt
"""

import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO


# ─── colour map for bounding box overlays ─────────────────────────────────────
PALETTE = [
    (255, 56,  56), (255, 157, 151), (255, 112,  31),
    (255, 178, 29),  (207, 210,  49), (72,  249, 10),
    (146, 204, 23),  (61,  219, 134), (26,  147, 52),
    (0,   212, 187),
]


def draw_detections(frame: np.ndarray, result, names: dict) -> np.ndarray:
    if result.boxes is None:
        return frame
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = PALETTE[cls_id % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cls_id]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def run_qcar(weights: str, conf: float, iou: float, show: bool):
    model = YOLO(weights)
    names = model.names
    print(f"Model loaded: {weights}")
    print(f"Classes: {names}")

    try:
        from qvl.qlabs import QuanserInteractiveLabs
        from qvl.qcar import QLabsQCar

        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        qcar = QLabsQCar(qlabs)
        use_qlab = True
        print("Connected to QLabs QCar")
    except ImportError:
        print("QLabs SDK not found — using webcam fallback")
        cap = cv2.VideoCapture(0)
        use_qlab = False

    fps_history = []
    try:
        while True:
            t0 = time.time()

            if use_qlab:
                success, frame = qcar.get_image(0)
                if not success:
                    time.sleep(0.05)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # Inference
            results = model(frame, conf=conf, iou=iou, verbose=False)
            result  = results[0]

            # Draw
            annotated = draw_detections(frame.copy(), result, names)

            dt = time.time() - t0
            fps = 1.0 / dt
            fps_history.append(fps)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Console output
            if result.boxes and len(result.boxes):
                for box in result.boxes:
                    cid = int(box.cls[0])
                    print(f"  Detected: {names[cid]} ({float(box.conf[0]):.2f})")

            if show:
                cv2.imshow("QCar Traffic Sign Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if use_qlab:
            qlabs.close()
        else:
            cap.release()
        cv2.destroyAllWindows()
        if fps_history:
            print(f"Average FPS: {sum(fps_history)/len(fps_history):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/runs/traffic_sign_v1/weights/best.pt")
    parser.add_argument("--conf",    type=float, default=0.4)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--show",    action="store_true", default=True)
    args = parser.parse_args()

    run_qcar(args.weights, args.conf, args.iou, args.show)
