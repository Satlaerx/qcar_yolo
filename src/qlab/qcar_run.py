"""src/qlab/qcar_run.py
键盘控制 QCar2 行驶，实时显示交通标志检测结果，可选同步采集图像。

用法（系统 Python，管理员权限）:
    D:\Python\python.exe src\qlab\collect_keyboard.py
    D:\Python\python.exe src\qlab\collect_keyboard.py --weights models\runs\traffic_sign_v1\weights\best.pt --no-setup

按键:
    W / ↑   前进        S / ↓   后退
    A / ←   左转        D / →   右转
    空格     停车        C       手动存图（按下沿触发，每次只存一张）
    Q        退出
"""

import sys
import argparse
import time
import threading
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO

from pal.products.qcar import QCar, IS_PHYSICAL_QCAR
from pal.utilities.vision import Camera2D

# ─── 控制参数 ──────────────────────────────────────────────────────────────────
THROTTLE_FWD = 0.15
THROTTLE_REV = -0.10
STEER_MAX    = 0.3
STEER_STEP   = 0.03
CTRL_HZ      = 50

CAM_ID  = "2@tcpip://localhost:18963"
CAM_W   = 820
CAM_H   = 410
CAM_FPS = 30

PALETTE = [
    (255,  56,  56),   # StopSign      红
    ( 72, 249,  10),   # YieldSign     绿
    (  0, 212, 187),   # RoundaboutSign 青
]


class KeyboardInference:
    def __init__(self, output_dir: Path, interval: float, show: bool,
                 auto_capture: bool, model: YOLO | None):
        self.output_dir   = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval     = interval
        self.show         = show
        self.auto_capture = auto_capture
        self.model        = model

        # 续接已有编号
        existing = []
        for p in self.output_dir.glob("frame_*.jpg"):
            try:
                existing.append(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        self.saved = max(existing) + 1 if existing else 0

        self.throttle      = 0.0
        self.steering      = 0.0
        self.manual_save   = False
        self.last_save_t   = 0.0
        self.running       = True
        self._c_prev_down  = False

    # ── 键盘线程 ──────────────────────────────────────────────────────────────
    def _key_loop(self):
        import keyboard
        print("\n键盘已就绪 — W/S/A/D 控制，空格停车，C 手动存图，Q 退出")
        if self.model:
            print("实时推理：已开启")
        if self.auto_capture:
            print(f"自动采集：已开启（每 {self.interval:.2f}s 一张）\n")
        else:
            print("自动采集：关闭（仅按 C 手动存图）\n")

        while self.running:
            if keyboard.is_pressed("w") or keyboard.is_pressed("up"):
                self.throttle = THROTTLE_FWD
            elif keyboard.is_pressed("s") or keyboard.is_pressed("down"):
                self.throttle = THROTTLE_REV
            else:
                self.throttle = 0.0

            if keyboard.is_pressed("a") or keyboard.is_pressed("left"):
                self.steering = min(self.steering + STEER_STEP, STEER_MAX)
            elif keyboard.is_pressed("d") or keyboard.is_pressed("right"):
                self.steering = max(self.steering - STEER_STEP, -STEER_MAX)
            else:
                if abs(self.steering) < STEER_STEP:
                    self.steering = 0.0
                else:
                    self.steering -= STEER_STEP * np.sign(self.steering)

            if keyboard.is_pressed("space"):
                self.throttle = 0.0
                self.steering = 0.0

            c_down = keyboard.is_pressed("c")
            if c_down and not self._c_prev_down:
                self.manual_save = True
            self._c_prev_down = c_down

            if keyboard.is_pressed("q"):
                self.running = False

            time.sleep(1 / CTRL_HZ)

    # ── 绘制检测框 ────────────────────────────────────────────────────────────
    def _draw_detections(self, frame, result):
        if result.boxes is None:
            return frame
        names = self.model.names
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

    # ── HUD ──────────────────────────────────────────────────────────────────
    def _draw_hud(self, frame, fps=None):
        cv2.putText(frame, f"Throttle: {self.throttle:+.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Steering: {self.steering:+.2f}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved:    {self.saved}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    # ── 主循环 ────────────────────────────────────────────────────────────────
    def run(self):
        threading.Thread(target=self._key_loop, daemon=True).start()

        print(f"初始化摄像头 {CAM_ID} ...")
        cam  = Camera2D(cameraId=CAM_ID, frameWidth=CAM_W,
                        frameHeight=CAM_H, frameRate=CAM_FPS)
        qcar = QCar(readMode=1, frequency=CTRL_HZ)

        with qcar:
            while self.running:
                t0 = time.time()
                qcar.write(self.throttle, self.steering)

                cam.read()
                frame = cam.imageData.copy()

                # 推理
                result = None
                if self.model:
                    results = self.model(frame, conf=0.4, iou=0.45, verbose=False)
                    result  = results[0]
                    frame   = self._draw_detections(frame, result)

                    # 控制台输出检测结果
                    if result.boxes and len(result.boxes):
                        for box in result.boxes:
                            cid = int(box.cls[0])
                            print(f"  检测到: {self.model.names[cid]}  "
                                  f"置信度: {float(box.conf[0]):.2f}")

                # 存图（存原始帧，不含检测框）
                t_now = time.time()
                raw   = cam.imageData
                need_auto = self.auto_capture and (t_now - self.last_save_t >= self.interval)
                if self.manual_save or need_auto:
                    fname = self.output_dir / f"frame_{self.saved:05d}.jpg"
                    cv2.imwrite(str(fname), raw)
                    self.saved       += 1
                    self.last_save_t  = t_now
                    self.manual_save  = False

                # 预览
                if self.show:
                    fps = 1.0 / max(time.time() - t0, 1e-6)
                    self._draw_hud(frame, fps)
                    cv2.imshow("QCar2  Drive + Detect  (Q=退出)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

                time.sleep(max(0, 1/CTRL_HZ - (time.time() - t0)))

            qcar.write(0.0, 0.0)

        cam.close()
        cv2.destroyAllWindows()
        print(f"\n✓ 结束。共保存 {self.saved} 张图像 → {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="data/raw")
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument("--weights",  default=None,
                        help="YOLO 权重路径，不传则只驾驶不推理")
    parser.add_argument("--no-show",  action="store_true")
    parser.add_argument("--no-setup", action="store_true")
    args = parser.parse_args()

    # 加载模型（可选）
    model = None
    if args.weights:
        print(f"加载模型: {args.weights}")
        model = YOLO(args.weights)
        print(f"类别: {model.names}\n")

    # 询问是否开启自动采集
    print("是否开启自动采集图像？[Y/n]: ", end="")
    choice = input().strip().lower()
    auto_capture = choice not in ("n", "no")

    # 初始化场景
    if not IS_PHYSICAL_QCAR and not args.no_setup:
        sys.path.insert(0, str(Path(__file__).parent))
        import setup_scene as qlabs_setup
        print("=== 初始化 QLabs 场景 ===")
        qlabs_setup.setup()
        print()

    collector = KeyboardInference(
        output_dir=Path(args.output),
        interval=args.interval,
        show=not args.no_show,
        auto_capture=auto_capture,
        model=model,
    )

    try:
        collector.run()
    except KeyboardInterrupt:
        collector.running = False
        print("\n用户中断")

    if not IS_PHYSICAL_QCAR and not args.no_setup:
        sys.path.insert(0, str(Path(__file__).parent))
        import setup_scene as qlabs_setup
        qlabs_setup.terminate()


if __name__ == "__main__":
    main()

