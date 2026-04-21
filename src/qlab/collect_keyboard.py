"""
src/qlab/collect_keyboard.py
键盘控制 QCar2 行驶，同时可选后台自动采集图像。

用法（系统 Python，管理员权限运行终端）:
    D:\Python\python.exe src\qlab\collect_keyboard.py
    D:\Python\python.exe src\qlab\collect_keyboard.py --output data\raw --interval 0.15 --no-setup

按键:
    W / ↑   前进        S / ↓   后退
    A / ←   左转        D / →   右转
    空格     停车        C       手动存图（按一次只保存一张）
    Q        退出
"""

import sys
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
    def __init__(self, output_dir: Path, interval: float, show: bool, auto_capture: bool):
        self.output_dir   = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.interval     = interval
        self.show         = show
        self.auto_capture = auto_capture

        self.throttle     = 0.0
        self.steering     = 0.0

        # 自动续接编号：从已有 frame_*.jpg 的最大编号 + 1 开始
        existing = []
        for p in self.output_dir.glob("frame_*.jpg"):
            try:
                existing.append(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        self.saved = max(existing) + 1 if existing else 0

        self.manual_save  = False
        self.last_save_t  = 0.0
        self.running      = True

        # 只对 C 键做防抖 / 按下沿触发
        self._c_prev_down = False

    def _key_loop(self):
        import keyboard

        print("\n键盘已就绪 — W/S/A/D 控制，空格停车，C 手动存图，Q 退出")
        if self.auto_capture:
            print(f"自动采集：开启（每 {self.interval:.3f}s 一张，约 {1.0 / self.interval:.2f} 张/秒）\n")
        else:
            print("自动采集：关闭（仅按 C 手动存图）\n")

        while self.running:
            # ── 车辆控制：保持原逻辑，不做防抖 ─────────────────────────────
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

            # ── C 键：按下沿触发，只保存一次 ────────────────────────────────
            c_down = keyboard.is_pressed("c")
            if c_down and not self._c_prev_down:
                self.manual_save = True
            self._c_prev_down = c_down

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

        auto_text = "ON" if self.auto_capture else "OFF"
        cv2.putText(frame, f"Auto Capture: {auto_text}",
                    (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    def _save_frame(self, frame, t_now):
        fname = self.output_dir / f"frame_{self.saved:05d}.jpg"
        cv2.imwrite(str(fname), frame)
        self.saved += 1
        self.last_save_t = t_now
        self.manual_save = False

    def run(self):
        threading.Thread(target=self._key_loop, daemon=True).start()

        print(f"初始化摄像头 {CAM_ID} ...")
        cam = Camera2D(
            cameraId=CAM_ID,
            frameWidth=CAM_W,
            frameHeight=CAM_H,
            frameRate=CAM_FPS
        )

        print("初始化 QCar 控制接口...")
        qcar = QCar(readMode=1, frequency=CTRL_HZ)

        if self.auto_capture:
            print(f"开始采集 → {self.output_dir}  （自动采集间隔 {self.interval:.3f}s）\n")
        else:
            print(f"开始采集 → {self.output_dir}  （仅手动存图）\n")

        with qcar:
            while self.running:
                t_now = time.time()
                qcar.write(self.throttle, self.steering)

                cam.read()
                frame = cam.imageData

                need_auto_save = self.auto_capture and ((t_now - self.last_save_t) >= self.interval)

                if self.manual_save or need_auto_save:
                    self._save_frame(frame, t_now)

                if self.show:
                    display = frame.copy()
                    self._draw_hud(display)
                    cv2.imshow("QCar Data Collection  (Q=退出)", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

                time.sleep(1 / CTRL_HZ)

            qcar.write(0.0, 0.0)

        cam.close()
        cv2.destroyAllWindows()
        print(f"\n✓ 采集完成。共保存 {self.saved} 张图像 → {self.output_dir}")


def ask_auto_capture(interval: float) -> bool:
    if interval <= 0:
        print("\n当前 --interval <= 0，自动采集无法启用，将自动切换为仅手动存图。\n")
        return False

    freq = 1.0 / interval
    print("\n================ 图像采集设置 ================")
    print(f"当前自动采集间隔: {interval:.3f} s/张")
    print(f"等效自动采集频率: {freq:.2f} 张/秒")
    print("是否开启自动采集？")
    print("  Y / y / 回车  -> 开启")
    print("  N / n         -> 关闭（仅按 C 手动存图）")
    print("=============================================\n")

    while True:
        choice = input("请输入 [Y/n]: ").strip().lower()
        if choice in ("", "y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print("输入无效，请输入 Y 或 N。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="data/raw")
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument("--no-show",  action="store_true")
    parser.add_argument("--no-setup", action="store_true",
                        help="跳过 QLabs 场景初始化（场景已就绪时使用）")
    args = parser.parse_args()

    # 运行前先让用户选择是否开启自动采集
    auto_capture = ask_auto_capture(args.interval)

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
        auto_capture=auto_capture,
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

