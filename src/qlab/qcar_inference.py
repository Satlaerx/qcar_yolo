"""src/qlab/qcar_inference.py
QCar2 实时交通标志检测推理。
使用与采集脚本相同的 Camera2D API 读取摄像头。

用法:
    D:\Python\python.exe src\qlab\qcar_inference.py --weights models\runs\traffic_sign_v1\weights\best.pt
    D:\Python\python.exe src\qlab\qcar_inference.py --weights models\runs\traffic_sign_v1\weights\best.pt --no-setup
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pal.utilities.vision import Camera2D
from pal.products.qcar import IS_PHYSICAL_QCAR

CAM_ID  = "2@tcpip://localhost:18963"
CAM_W   = 820
CAM_H   = 410
CAM_FPS = 30

PALETTE = [
    (255,  56,  56),
    ( 72, 249,  10),
    (  0, 212, 187),
]


def draw_detections(frame, result, names):
    if result.boxes is None:
        return frame
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = PALETTE[cls_id % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cls_id]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def run(weights: str, conf: float, iou: float, show: bool):
    model = YOLO(weights)
    names = model.names
    print(f"模型加载完成: {weights}")
    print(f"类别: {names}")

    print(f"初始化摄像头 {CAM_ID} ...")
    cam = Camera2D(cameraId=CAM_ID, frameWidth=CAM_W,
                   frameHeight=CAM_H, frameRate=CAM_FPS)
    print("摄像头就绪，开始推理（按 Q 退出）\n")

    fps_list = []
    try:
        while True:
            t0 = time.time()

            cam.read()
            frame = cam.imageData

            results = model(frame, conf=conf, iou=iou, verbose=False)
            result  = results[0]

            annotated = draw_detections(frame.copy(), result, names)

            fps = 1.0 / max(time.time() - t0, 1e-6)
            fps_list.append(fps)
            cv2.putText(annotated, f"FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            if result.boxes and len(result.boxes):
                for box in result.boxes:
                    cid = int(box.cls[0])
                    print(f"  检测到: {names[cid]}  置信度: {float(box.conf[0]):.2f}")

            if show:
                cv2.imshow("QCar2 Traffic Sign Detection  (Q=退出)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if fps_list:
            print(f"平均 FPS: {sum(fps_list)/len(fps_list):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  default="models/runs/traffic_sign_v1/weights/best.pt")
    parser.add_argument("--conf",     type=float, default=0.4)
    parser.add_argument("--iou",      type=float, default=0.45)
    parser.add_argument("--no-show",  action="store_true")
    parser.add_argument("--no-setup", action="store_true",
                        help="跳过场景初始化（场景已就绪时）")
    args = parser.parse_args()

    if not IS_PHYSICAL_QCAR and not args.no_setup:
        sys.path.insert(0, str(Path(__file__).parent))
        import setup_scene as qlabs_setup
        print("=== 初始化 QLabs 场景 ===")
        qlabs_setup.setup()
        print()

    run(args.weights, args.conf, args.iou, not args.no_show)


if __name__ == "__main__":
    main()