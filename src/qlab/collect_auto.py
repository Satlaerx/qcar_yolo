"""src/qlab/collect_keyboard.py
基于官方路网的自动驾驶采集脚本。

完全照搬官方 environment_interpretation.py 的控制架构：
  - QCarDriveController  → 官方封装的速度+Stanley转向一体控制器
  - QCarEKF              → 状态估计
  - QCarGPS              → 传感器
  - Camera2D             → 摄像头

图片保存到: data/raw/frame_XXXXX.jpg

用法:
    D:\Python\python.exe src\qlab\collect_auto.py
    D:\Python\python.exe src\qlab\collect_auto.py --loops 3 --no-setup
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import cv2

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.vision import Camera2D
from hal.content.qcar_functions import QCarEKF, QCarDriveController
from hal.products.mats import SDCSRoadMap

# ─── 参数 ──────────────────────────────────────────────────────────────────────
CTRL_HZ       = 100
V_REF         = 0.3    # 巡逻速度 m/s，慢速保证图像清晰
INTERVAL      = 0.12   # 存图间隔秒（约 8fps）
START_DELAY   = 1.0    # 等 EKF 稳定的秒数
CAM_ID        = "2@tcpip://localhost:18963"
CAM_W, CAM_H, CAM_FPS = 820, 410, 30

# 路径节点序列（来自官方 environment_interpretation.py）
# cyclic=False 表示走完一遍算一圈，不自动循环（我们在外层控制圈数）
NODE_SEQUENCE = [0, 20, 0]


class AutoCollector:
    def __init__(self, output_dir: Path, loops: int, show: bool):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loops   = loops
        self.show    = show
        self.saved   = 0
        self.running = True

    def _run_one_lap(self, qcar, cam, gps, ekf, initial_pose):
        """跑一圈路径，返回本圈采集帧数。"""
        # 每圈重新创建 DriveController，路径从头开始
        roadmap   = SDCSRoadMap(leftHandTraffic=False)
        waypoints = roadmap.generate_path(NODE_SEQUENCE)
        ctrl      = QCarDriveController(waypoints, cyclic=False)

        last_save = 0.0
        lap_saved = 0
        u, delta  = 0.0, 0.0

        t0 = time.time()
        t  = 0.0

        while self.running:
            tp = t
            t  = time.time() - t0
            dt = max(t - tp, 1e-4)

            # 读传感器
            qcar.read()
            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2],
                ])
                ekf.update([qcar.motorTach, delta], dt, y_gps, qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, delta], dt, None, qcar.gyroscope[2])

            p  = np.array([ekf.x_hat[0, 0], ekf.x_hat[1, 0]])
            th = ekf.x_hat[2, 0]
            v  = qcar.motorTach

            # 官方 QCarDriveController.update(p, th, v, v_ref, dt)
            if t < START_DELAY:
                u, delta = 0.0, 0.0
            else:
                u, delta = ctrl.update(p, th, v, V_REF, dt)

            qcar.write(u, delta)

            # 存图
            cam.read()
            frame  = cam.imageData
            t_now  = time.time()
            if t_now - last_save >= INTERVAL:
                cv2.imwrite(
                    str(self.output_dir / f"frame_{self.saved:05d}.jpg"),
                    frame)
                self.saved   += 1
                lap_saved    += 1
                last_save     = t_now

            # 预览
            if self.show:
                disp = frame.copy()
                cv2.putText(disp, f"u={u:.2f} steer={delta:.2f} saved={self.saved}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.imshow("Auto Collect (Ctrl+C 停止)", disp)
                cv2.waitKey(1)

            # 路径走完时退出本圈
            if ctrl.steeringController.pathComplete:
                break

        return lap_saved

    def run(self):
        roadmap      = SDCSRoadMap(leftHandTraffic=False)
        initial_pose = roadmap.get_node_pose(NODE_SEQUENCE[0]).squeeze()

        ekf = QCarEKF(x_0=initial_pose)
        gps = QCarGPS(initialPose=initial_pose, calibrate=False)
        while gps.readGPS():
            pass

        cam  = Camera2D(cameraId=CAM_ID, frameWidth=CAM_W,
                        frameHeight=CAM_H, frameRate=CAM_FPS)
        qcar = QCar(readMode=1, frequency=CTRL_HZ)

        print(f"\n开始自动驾驶采集")
        print(f"  路径节点 : {NODE_SEQUENCE}")
        print(f"  巡逻圈数 : {self.loops} 圈")
        print(f"  巡逻速度 : {V_REF} m/s")
        print(f"  保存目录 : {self.output_dir.resolve()}")
        print(f"  Ctrl+C 安全停止\n")

        try:
            with qcar:
                for lap in range(1, self.loops + 1):
                    if not self.running:
                        break
                    print(f"── 第 {lap}/{self.loops} 圈 ──────────────────────")
                    n = self._run_one_lap(qcar, cam, gps, ekf, initial_pose)
                    print(f"  本圈采集 {n} 张，累计 {self.saved} 张")

                qcar.write(0.0, 0.0)

        except KeyboardInterrupt:
            pass

        cam.close()
        cv2.destroyAllWindows()
        print(f"\n✓ 采集完成！共 {self.saved} 张 → {self.output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="data/raw")
    parser.add_argument("--loops",    type=int, default=5)
    parser.add_argument("--no-show",  action="store_true")
    parser.add_argument("--no-setup", action="store_true")
    args = parser.parse_args()

    if not IS_PHYSICAL_QCAR and not args.no_setup:
        sys.path.insert(0, str(Path(__file__).parent))
        import setup_scene as qlabs_setup
        roadmap      = SDCSRoadMap(leftHandTraffic=False)
        initial_pose = roadmap.get_node_pose(NODE_SEQUENCE[0]).squeeze()
        qlabs_setup.setup(
            initial_position=[initial_pose[0], initial_pose[1], 0.0],
            initial_orientation=[0, 0, initial_pose[2]],
        )
        print()

    collector = AutoCollector(
        output_dir=Path(args.output),
        loops=args.loops,
        show=not args.no_show,
    )

    try:
        collector.run()
    except KeyboardInterrupt:
        collector.running = False
        print(f"\n已中断，共保存 {collector.saved} 张")

    if not IS_PHYSICAL_QCAR and not args.no_setup:
        sys.path.insert(0, str(Path(__file__).parent))
        import setup_scene as qlabs_setup
        qlabs_setup.terminate()


if __name__ == "__main__":
    main()