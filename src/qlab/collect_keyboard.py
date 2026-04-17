"""src/qlab/collect_keyboard.py
键盘控制 QCar2 行驶，同时后台自动采集图像。
基于官方示例的真实 API：
  - QCar（pal.products.qcar）控制油门/转向
  - Camera2D（pal.utilities.vision）读取摄像头

用法（系统 Python，管理员权限运行终端）:
    D:\Python\python.exe src\qlab\collect_keyboard.py
    D:\Python\python.exe src\qlab\collect_keyboard.py --output data\raw --interval 0.15

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

# Quanser 真实 API
from pal.products.qcar import QCar, IS_PHYSICAL_QCAR
from pal.utilities.vision import Camera2D

# 场景初始化（仿真模式下先 setup）
import src.qlab.setup_scene as qlabs_setup   # 或直接 import setup_scene


# ─── 控制参数 ──────────────────────────────────────────────────────────────────
THROTTLE_FWD  =  0.15   # 前进油门（官方示例用 0.3，采集时开慢点）
THROTTLE_REV  = -0.10
STEER_MAX     =  0.3    # 最大转向（弧度）
STEER_STEP    =  0.03   # 每帧转向增量（平滑回正）
CTRL_HZ       =  50     # 控制循环频率

# 摄像头参数（与官方示例 QCar2_Color_Space.py 一致）
CAM_ID     = "2@tcpip://localhost:18963"  # QLabs CSI 前置摄像头
CAM_W      = 820    # 采集时用一半分辨率，加快速度
CAM_H      = 410
CAM_FPS    = 30

# 全局停止标志（Ctrl+C 触发）
KILL = False
def _sig_handler(*_):
    global KILL
    KILL = True
signal.signal(signal.SIGINT, _sig_handler)


class KeyboardCollector:
    def __init__(self, output_dir: Path, interval: float, show: bool):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval    = interval
        self.show        = show

        self.throttle    = 0.0
        self.steering    = 0.0
        self.saved       = 0
        self.manual_save = False
        self.last_save_t = 0.0

    # ── 键盘轮询线程 ──────────────────────────────────────────────────────────
    def _key_loop(self):
        import keyboard
        print("\n键盘已就绪 — W/S/A/D 控制，空格停车，C 手动存图，Q 退出\n")
        while not KILL:
            # 纵向
            if keyboard.is_pressed("w") or keyboard.is_pressed("up"):
                self.throttle = THROTTLE_FWD
            elif keyboard.is_pressed("s") or keyboard.is_pressed("down"):
                self.throttle = THROTTLE_REV
            else:
                self.throttle = 0.0

            # 横向平滑
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
                global KILL
                KILL = True

            time.sleep(1 / CTRL_HZ)

    # ── HUD 叠加 ──────────────────────────────────────────────────────────────
    def _draw_hud(self, frame):
        cv2.putText(frame, f"Throttle: {self.throttle:+.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Steering: {self.steering:+.2f}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Saved: {self.saved}",
                    (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

    # ── 主循环 ────────────────────────────────────────────────────────────────
    def run(self):
        # 启动键盘线程
        threading.Thread(target=self._key_loop, daemon=True).start()

        # 初始化摄像头（Camera2D，与官方示例相同）
        print(f"初始化摄像头 {CAM_ID} ...")
        cam = Camera2D(
            cameraId=CAM_ID,
            frameWidth=CAM_W,
            frameHeight=CAM_H,
            frameRate=CAM_FPS,
        )

        # 初始化 QCar 控制接口
        print("初始化 QCar 控制接口...")
        qcar = QCar(readMode=1, frequency=CTRL_HZ)

        print(f"开始采集 → {self.output_dir}  (间隔 {self.interval}s)\n")

        with qcar:
            while not KILL:
                t_now = time.time()

                # 写入控制指令
                qcar.write(self.throttle, self.steering)

                # 读取摄像头
                cam.read()
                frame = cam.imageData   # numpy BGR

                # 判断是否保存
                if self.manual_save or (t_now - self.last_save_t) >= self.interval:
                    fname = self.output_dir / f"frame_{self.saved:05d}.jpg"
                    cv2.imwrite(str(fname), frame)
                    self.saved += 1
                    self.last_save_t = t_now
                    self.manual_save = False

                # 预览
                if self.show:
                    display = frame.copy()
                    self._draw_hud(display)
                    cv2.imshow("QCar Data Collection  (Q=退出)", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        global KILL
                        KILL = True

                time.sleep(1 / CTRL_HZ)

            # 停车
            qcar.write(0.0, 0.0)

        cam.close()
        cv2.destroyAllWindows()
        print(f"\n✓ 采集完成。共保存 {self.saved} 张图像 → {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="data/raw",
                        help="图像保存目录（默认 data/raw）")
    parser.add_argument("--interval", type=float, default=0.15,
                        help="自动存图间隔秒数（默认 0.15 ≈ 6fps）")
    parser.add_argument("--no-show",  action="store_true",
                        help="不显示预览窗口（无 GUI 环境）")
    parser.add_argument("--no-setup", action="store_true",
                        help="跳过 QLabs 场景初始化（场景已就绪时使用）")
    args = parser.parse_args()

    # 仿真模式下先布置场景
    if not IS_PHYSICAL_QCAR and not args.no_setup:
        print("=== 初始化 QLabs 场景 ===")
        qlabs_setup.setup()
        print()

    collector = KeyboardCollector(
        output_dir=Path(args.output),
        interval=args.interval,
        show=not args.no_show,
    )
    collector.run()

    # 退出时清理场景
    if not IS_PHYSICAL_QCAR and not args.no_setup:
        qlabs_setup.terminate()


if __name__ == "__main__":
    main()
