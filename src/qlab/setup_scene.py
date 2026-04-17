"""src/qlab/setup_scene.py
在 QLabs 场景里批量放置交通标志。
运行一次即可，保存 QLabs 场景后标志永久存在。

用法:
    python src/qlab/setup_scene.py
    python src/qlab/setup_scene.py --clear   # 先清除已有标志再重新放
"""

import argparse
from qvl.qlabs import QuanserInteractiveLabs
from qvl.traffic_sign import QLabsTrafficSign

# ─── 交通标志布局配置 ──────────────────────────────────────────────────────────
# 每条记录: (x, y, z, yaw度, sign_type, 备注)
# sign_type 参考 QLabsTrafficSign 的常量，例如:
#   SIGN_STOP / SIGN_YIELD / SIGN_NO_ENTRY / SIGN_SPEED_LIMIT_* 等
# 如果你的 SDK 版本常量名不同，改成对应整数 ID 即可

SIGN_LAYOUT = [
    # ── 直道段：QCar 正前方，不同距离 ──────────────────────────────
    ( 3.0,  0.0, 0.0,   0, "SIGN_STOP",          "近距离正对"),
    ( 6.0,  0.0, 0.0,   0, "SIGN_SPEED_LIMIT_30","中距离正对"),
    (10.0,  0.0, 0.0,   0, "SIGN_YIELD",         "远距离正对"),

    # ── 路侧：右侧偏角 ──────────────────────────────────────────────
    ( 4.0, -1.2, 0.0, -20, "SIGN_NO_ENTRY",      "右侧-20度"),
    ( 7.0, -1.5, 0.0, -35, "SIGN_SPEED_LIMIT_50","右侧-35度"),

    # ── 路侧：左侧偏角 ──────────────────────────────────────────────
    ( 5.0,  1.2, 0.0,  20, "SIGN_TURN_RIGHT",    "左侧+20度"),
    ( 8.0,  1.5, 0.0,  35, "SIGN_PEDESTRIAN",    "左侧+35度"),

    # ── 弯道处 ─────────────────────────────────────────────────────
    (12.0,  2.0, 0.0,  45, "SIGN_TURN_LEFT",     "弯道前"),
    (15.0, -1.0, 0.0, -30, "SIGN_SPEED_LIMIT_80","弯道后"),
]

# sign_type 字符串 → QLabsTrafficSign 常量映射
# 根据你的 SDK 版本调整，不确定时直接用整数 ID
SIGN_TYPE_MAP = {
    "SIGN_STOP":           QLabsTrafficSign.SIGN_STOP           if hasattr(QLabsTrafficSign, "SIGN_STOP")           else 0,
    "SIGN_YIELD":          QLabsTrafficSign.SIGN_YIELD          if hasattr(QLabsTrafficSign, "SIGN_YIELD")          else 1,
    "SIGN_SPEED_LIMIT_30": QLabsTrafficSign.SIGN_SPEED_LIMIT_30 if hasattr(QLabsTrafficSign, "SIGN_SPEED_LIMIT_30") else 2,
    "SIGN_SPEED_LIMIT_50": QLabsTrafficSign.SIGN_SPEED_LIMIT_50 if hasattr(QLabsTrafficSign, "SIGN_SPEED_LIMIT_50") else 3,
    "SIGN_SPEED_LIMIT_80": QLabsTrafficSign.SIGN_SPEED_LIMIT_80 if hasattr(QLabsTrafficSign, "SIGN_SPEED_LIMIT_80") else 4,
    "SIGN_NO_ENTRY":       QLabsTrafficSign.SIGN_NO_ENTRY       if hasattr(QLabsTrafficSign, "SIGN_NO_ENTRY")       else 5,
    "SIGN_TURN_LEFT":      QLabsTrafficSign.SIGN_TURN_LEFT      if hasattr(QLabsTrafficSign, "SIGN_TURN_LEFT")      else 6,
    "SIGN_TURN_RIGHT":     QLabsTrafficSign.SIGN_TURN_RIGHT     if hasattr(QLabsTrafficSign, "SIGN_TURN_RIGHT")     else 7,
    "SIGN_PEDESTRIAN":     QLabsTrafficSign.SIGN_PEDESTRIAN     if hasattr(QLabsTrafficSign, "SIGN_PEDESTRIAN")     else 8,
}


def setup_scene(clear: bool = False):
    import math

    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    qlabs.open("localhost")

    sign_actor = QLabsTrafficSign(qlabs)

    if clear:
        print("Clearing existing traffic signs...")
        sign_actor.destroy_all_instances()

    print(f"Spawning {len(SIGN_LAYOUT)} traffic signs...")
    for i, (x, y, z, yaw_deg, sign_type_str, note) in enumerate(SIGN_LAYOUT):
        yaw_rad = math.radians(yaw_deg)
        sign_type_id = SIGN_TYPE_MAP.get(sign_type_str, 0)

        ret = sign_actor.spawn_id(
            actorNumber=i,
            location=[x, y, z],
            rotation=[0, 0, yaw_rad],
            scale=[1, 1, 1],
            configuration=sign_type_id,
            waitForConfirmation=True,
        )

        status = "OK" if ret == 0 else f"WARN(ret={ret})"
        print(f"  [{status}] #{i:02d} {sign_type_str:25s} @ ({x:5.1f},{y:5.1f}) yaw={yaw_deg:+d}°  {note}")

    print("\n场景布置完成。请在 QLabs 里 File → Save Scene 保存。")
    qlabs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true",
                        help="先清除场景中所有已有交通标志")
    args = parser.parse_args()
    setup_scene(clear=args.clear)
