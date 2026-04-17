"""src/qlab/setup_scene.py
在 QLabs 场景里初始化 QCar2 并布置交通标志。
基于 Quanser 官方示例代码（qlabs_setup.py）的实际 API 编写。

用法（系统 Python，不要用 venv）:
    D:\Python\python.exe src\qlab\setup_scene.py
    D:\Python\python.exe src\qlab\setup_scene.py --clear
"""

import os
import sys
import argparse
import numpy as np
import time

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.system import QLabsSystem

# 每种标志是独立的类，不存在通用 QLabsTrafficSign
from qvl.stop_sign import QLabsStopSign
from qvl.yield_sign import QLabsYieldSign
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.traffic_light import QLabsTrafficLight

import pal.resources.rtmodels as rtmodels


# ─── 交通标志布局配置（Cityscape 场景坐标）────────────────────────────────────
# 格式: (location[x,y,z], rotation[roll,pitch,yaw_rad], 备注)

STOP_SIGNS = [
    ([-0.508, -7.327, 0.2], [0, 0, np.pi/2],  "路口停车线"),
    ([ 5.0,    3.0,   0.2], [0, 0, np.pi],    "直道右侧"),
    ([ 8.0,   -2.0,  0.2],  [0, 0, np.pi/4],  "弯道前"),
]

YIELD_SIGNS = [
    ([0.4,  -13.0, 0.0], [0, 0, np.pi],   "路口让行"),
    ([6.0,    5.0, 0.0], [0, 0, np.pi/2], "交叉路口"),
]

ROUNDABOUT_SIGNS = [
    ([24.5, 33.0, 0.0], [0, 0, -np.pi/2], "环岛入口1"),
    ([ 4.5, 40.0, 0.0], [0, 0,  np.pi],   "环岛入口2"),
    ([10.6, 28.5, 0.0], [0, 0,  np.pi],   "环岛入口3"),
]


def setup(
    initial_position=[1.36, 1.311, 0.0],
    initial_orientation=[0, 0, -np.pi/2],
    rt_model=rtmodels.QCAR2,
    clear=False,
):
    os.system('cls')
    qlabs = QuanserInteractiveLabs()

    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("无法连接到 QLabs，请确认 QLabs 已启动并加载了场景")
        sys.exit(1)
    print("Connected to QLabs")

    # 无论是否 --clear，都先停止旧实时模型，避免"lacks privileges"报错
    print("停止旧实时模型（如有）...")
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(0.5)

    if clear:
        print("清除场景中所有已有 Actor...")
        qlabs.destroy_all_spawned_actors()
        time.sleep(0.3)

    QLabsSystem(qlabs).set_title_string('Traffic Sign Detection - Data Collection')

    # 生成 QCar2
    print(f"生成 QCar2 @ {initial_position} ...")
    hqcar = QLabsQCar2(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=initial_position,
        rotation=initial_orientation,
        waitForConfirmation=True,
    )

    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn([8.484, 1.973, 12.209], [0, 0.748, 0.792])
    hqcar.possess()

    # 生成各类交通标志
    print("\n生成停车标志...")
    for i, (loc, rot, note) in enumerate(STOP_SIGNS):
        sign = QLabsStopSign(qlabs)
        ret = sign.spawn(location=loc, rotation=rot,
                         scale=[1,1,1], configuration=0,
                         waitForConfirmation=True)
        print(f"  [{'OK' if ret==0 else f'WARN ret={ret}'}] StopSign #{i}  {note}")

    print("生成让行标志...")
    for i, (loc, rot, note) in enumerate(YIELD_SIGNS):
        sign = QLabsYieldSign(qlabs)
        ret = sign.spawn(location=loc, rotation=rot, waitForConfirmation=True)
        print(f"  [{'OK' if ret==0 else f'WARN ret={ret}'}] YieldSign #{i}  {note}")

    print("生成环岛标志...")
    for i, (loc, rot, note) in enumerate(ROUNDABOUT_SIGNS):
        sign = QLabsRoundaboutSign(qlabs)
        ret = sign.spawn(location=loc, rotation=rot, waitForConfirmation=True)
        print(f"  [{'OK' if ret==0 else f'WARN ret={ret}'}] RoundaboutSign #{i}  {note}")

    # 启动实时模型（必须，否则 QCar 物理不生效）
    print(f"\n启动实时模型...")
    QLabsRealTime().start_real_time_model(rt_model)

    print("\n✓ 场景布置完成！")
    print("  提示: 在 QLabs 中 File → Save Scene 可保存布局")
    qlabs.close()
    return hqcar


def terminate():
    """实验结束时清理场景，可在其他脚本里调用。"""
    qlabs = QuanserInteractiveLabs()
    if qlabs.open("localhost"):
        qlabs.destroy_all_spawned_actors()
        QLabsRealTime().terminate_all_real_time_models()
        qlabs.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true",
                        help="先清除场景中所有已有 Actor 再重新生成")
    parser.add_argument("--x",   type=float, default=1.36,
                        help="QCar 初始 X 坐标（默认 Cityscape 起点）")
    parser.add_argument("--y",   type=float, default=1.311,
                        help="QCar 初始 Y 坐标")
    parser.add_argument("--yaw", type=float, default=-90,
                        help="QCar 初始朝向（度，默认 -90）")
    args = parser.parse_args()

    setup(
        initial_position=[args.x, args.y, 0.0],
        initial_orientation=[0, 0, np.radians(args.yaw)],
        clear=args.clear,
    )