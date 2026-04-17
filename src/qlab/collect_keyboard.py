"""src/qlab/collect_keyboard.py
用键盘实时控制 QCar，同时后台自动采集图像。

按键说明:
    W / ↑   前进
    S / ↓   后退
    A / ←   左转
    D / →   右转
    空格     刹车 / 停车
    C        手动保存当前帧（不管自动间隔）
    Q        退出并保存统计

用法:
    pip install keyboard         # 需要管理员权限运行
    python src/qlab/collect_keyboard.py
    python src/qlab/collect_keyboard.py --output data/raw --interval 0.15 --show
"""

import argparse
import time
import threading
from pathlib import Path

import cv2
import numpy as np

# ─── 控制参数 ──────────────────────────────────────────────────────────────────
THROTTLE_FWD  =  0.25   # 前进油门 (0~1)
THROTTLE_REV  = -0.20   # 后退油门
STEER_MAX     =  0.35   # 最大转向角 (弧度近似)
STEER_STEP    =  0.05   # 每帧转向增量（平滑）


class KeyboardCollector:
    def __init__(self, output_dir: Path, interval: float, show: bool):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval   = interval
        self.show       = show

        self.throttle   = 0.0
        self.steering   = 0.0
        self.running    = True
        self.saved      = 0
        self.manual_save = False
        self.last_save_t = 0.0

        # 连接 QLabs
        from qvl.qlabs import QuanserInteractiveLabs
        from qvl.qcar import QLabsQCar
        self.qlabs = QuanserInteractiveLabs()
        self.qlabs.open("localhost")
        self.qcar  = QLabsQCar(self.qlabs)
        print("Connected to QLabs QCar.")

    # ── 键盘轮询（不依赖 GUI 焦点） ──────────────────────────────────────────
    def _read_keys(self):
        """在独立线程里持续轮询按键状态。"""
        import keyboard  # pip install keyboard (需管理员)
        print("\n键盘控制已启动 — W/S/A/D 控制，空格刹车，C 手动存图，Q 退出\n")

        while self.running:
            # 纵向
            if keyboard.is_pressed("w") or keyboard.is_pressed("up"):
                self.throttle = THROTTLE_FWD
            elif keyboard.is_pressed("s") or keyboard.is_pressed("down"):
                self.throttle = THROTTLE_REV
            else:
                self.throttle = 0.0

            # 横向（平滑）
            if keyboard.is_pressed("a") or keyboard.is_pressed("left"):
                self.steering = min(self.steering + STEER_STEP, STEER_MAX)
            elif keyboard.is_pressed("d") or keyboard.is_pressed("right"):
                self.steering = max(self.steering - STEER_STEP, -STEER_MAX)
            else:
                # 回正
                if abs(self.steering) < STEER_STEP:
                    self.steering = 0.0
                else:
                    self.steering -= STEER_STEP * (1 if self.steering > 0 else -1)

            # 刹车
            if keyboard.is_pressed("space"):
                self.throttle = 0.0
                self.steering = 0.0

            # 手动存图
            if keyboard.is_pressed("c"):
                self.manual_save = True

            # 退出
            if keyboard.is_pressed("q"):
                self.running = False

            time.sleep(0.03)   # ~33 Hz 轮询

    # ── 主循环 ────────────────────────────────────────────────────────────────
    def run(self):
        key_thread = threading.Thread(target=self._read_keys, daemon=True)
        key_thread.start()

        print(f"自动存图间隔: {self.interval}s  →  输出目录: {self.output_dir}")

        try:
            while self.running:
                t_now = time.time()

                # 发送控制指令
                self.qcar.set_velocity_and_request_state(
                    speed=self.throttle,
                    steering=self.steering,
                    headlights=True,
                    leftTurnSignal=(self.steering > 0.1),
                    rightTurnSignal=(self.steering < -0.1),
                    brakeSignal=(self.throttle == 0.0),
                    reverseSignal=(self.throttle < 0),
                )

                # 读取摄像头
                ok, frame = self.qcar.get_image(0)  # 0 = 前置摄像头
                if not ok:
                    time.sleep(0.05)
                    continue

                # 判断是否保存
                should_save = (
                    self.manual_save or
                    (t_now - self.last_save_t) >= self.interval
                )
                if should_save:
                    fname = self.output_dir / f"frame_{self.saved:05d}.jpg"
                    cv2.imwrite(str(fname), frame)
                    self.saved += 1
                    self.last_save_t = t_now
                    self.manual_save = False

                # 实时预览（叠加 HUD）
                if self.show:
                    display = frame.copy()
                    self._draw_hud(display)
                    cv2.imshow("QCar — 按 Q 退出", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

                time.sleep(0.02)   # ~50 Hz 主循环

        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _draw_hud(self, frame: np.ndarray):
        """在预览窗口叠加速度/转向/帧数信息。"""
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Throttle: {self.throttle:+.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Steering: {self.steering:+.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved:    {self.saved}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # 简单方向盘示意
        cx, cy, r = w - 60, 60, 40
        cv2.circle(frame, (cx, cy), r, (200, 200, 200), 2)
        sx = int(cx + r * 0.8 * (-self.steering / STEER_MAX))
        cv2.line(frame, (cx, cy), (sx, cy + 20), (0, 255, 255), 3)

    def _cleanup(self):
        self.qcar.set_velocity_and_request_state(
            speed=0, steering=0,
            headlights=False, leftTurnSignal=False,
            rightTurnSignal=False, brakeSignal=True, reverseSignal=False,
        )
        self.qlabs.close()
        cv2.destroyAllWindows()
        print(f"\n采集完成。共保存 {self.saved} 张图像 → {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="data/raw",
                        help="图像保存目录")
    parser.add_argument("--interval", type=float, default=0.15,
                        help="自动存图间隔（秒），默认 0.15s ≈ 6~7fps")
    parser.add_argument("--show",     action="store_true", default=True,
                        help="显示实时预览窗口")
    args = parser.parse_args()

    collector = KeyboardCollector(
        output_dir=Path(args.output),
        interval=args.interval,
        show=args.show,
    )
    collector.run()


if __name__ == "__main__":
    main()
