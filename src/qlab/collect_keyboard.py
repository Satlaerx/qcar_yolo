"""src/qlab/collect_keyboard.py
键盘控制 QCar2 行驶，同时后台自动采集图像。

用法（系统 Python，管理员权限运行终端）:
    D:\Python\python.exe src\qlab\collect_keyboard.py
    D:\Python\python.exe src\qlab\collect_keyboard.py --output data\raw --interval 0.15 --no-setup

按键:
    W / ↑   前进        S / ↓   后退
    A / ←   左转        D / →   右转
    空格     停车        C       手动存图
    Q        退出
"""

import sys
import signal
import argparse
import time
import threading
from pathlib import Path

import numpy as np
import cv2

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


class KeyboardCollector:
    def __init__(self, output_dir: Path, interval: float, show: bool):
        self.output_dir  = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval    = interval
        self.show        = show
        self.throttle    = 0.0
        self.steering    = 0.0
        self.saved       = 0
        self.manual_save = False
        self.last_save_t = 0.0
        self.running     = True  # 用实例属性，避免 global 语法问题

    def _key_loop(self):
        import keyboard
        print("\n键盘已就绪 — W/S/A/D 控制，空格停车，C 手动存图，Q 退出\n")
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

            if keyboard.is_pressed("c"):
                self.manual_save = True

            if keyboard.is_pressed("q"):
                self.running = False

            time.sleep(1 / CTRL_HZ)

    def _draw_hud(self, frame):
        cv2.putText(frame, f"Throttle: {self.throttle:+.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Steering: {self.steering:+.2f}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved:    {self.saved}",
                    (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    def run(self):
        threading.Thread(target=self._key_loop, daemon=True).start()

        print(f"初始化摄像头 {CAM_ID} ...")
        cam = Camera2D(cameraId=CAM_ID, frameWidth=CAM_W,
                       frameHeight=CAM_H, frameRate=CAM_FPS)

        print("初始化 QCar 控制接口...")
        qcar = QCar(readMode=1, frequency=CTRL_HZ)

        print(f"开始采集 → {self.output_dir}  (间隔 {self.interval}s)\n")

        with qcar:
            while self.running:
                t_now = time.time()
                qcar.write(self.throttle, self.steering)

                cam.read()
                frame = cam.imageData

                if self.manual_save or (t_now - self.last_save_t) >= self.interval:
                    fname = self.output_dir / f"frame_{self.saved:05d}.jpg"
                    cv2.imwrite(str(fname), frame)
                    self.saved += 1
                    self.last_save_t = t_now
                    self.manual_save = False

                if self.show:
                    display = frame.copy()
                    self._draw_hud(display)
                    cv2.imshow("QCar Data Collection  (Q=退出)", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                time.sleep(1 / CTRL_HZ)

            qcar.write(0.0, 0.0)

        cam.close()
        cv2.destroyAllWindows()
        print(f"\n✓ 采集完成。共保存 {self.saved} 张图像 → {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="data/raw")
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument("--no-show",  action="store_true")
    parser.add_argument("--no-setup", action="store_true",
                        help="跳过 QLabs 场景初始化（场景已就绪时使用）")
    args = parser.parse_args()

    if not IS_PHYSICAL_QCAR and not args.no_setup:
        sys.path.insert(0, str(Path(__file__).parent))
        import setup_scene as qlabs_setup
        print("=== 初始化 QLabs 场景 ===")
        qlabs_setup.setup()
        print()

    collector = KeyboardCollector(
        output_dir=Path(args.output),
        interval=args.interval,
        show=not args.no_show,
    )

    try:
        collector.run()
    except KeyboardInterrupt:
        collector.running = False
        print("\n用户中断")

    if not IS_PHYSICAL_QCAR and not args.no_setup:
        qlabs_setup.terminate()


if __name__ == "__main__":
    main()