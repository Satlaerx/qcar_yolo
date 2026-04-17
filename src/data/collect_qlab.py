"""src/data/collect_qlab.py
Capture images from the QCar front camera in QLabs simulation.
Saves frames to data/raw/ for annotation.

Requires:
  - QLabs installed and running
  - Quanser Python bindings on PATH
"""

import argparse
import time
from pathlib import Path
import cv2


def collect_from_qlab(output_dir: Path, num_frames: int, interval: float):
    """Stream frames from QLabs QCar camera and save to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Quanser QLabs imports (installed with QLabs SDK)
        from qvl.qlabs import QuanserInteractiveLabs
        from qvl.qcar import QLabsQCar

        qlabs = QuanserInteractiveLabs()
        print("Connecting to QLabs...")
        qlabs.open("localhost")

        qcar = QLabsQCar(qlabs)

        print(f"Collecting {num_frames} frames → {output_dir}")
        saved = 0
        while saved < num_frames:
            # Read CSI camera image (front-facing, channel 0)
            success, image = qcar.get_image(0)  # returns (bool, np.ndarray BGR)
            if not success:
                print("  Warning: camera read failed, retrying...")
                time.sleep(0.1)
                continue

            fname = output_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(fname), image)
            saved += 1
            if saved % 50 == 0:
                print(f"  Saved {saved}/{num_frames}")
            time.sleep(interval)

        qlabs.close()
        print(f"Done. {saved} frames saved to {output_dir}")

    except ImportError:
        # Fallback: collect from webcam for testing without QLabs
        print("QLabs SDK not found — falling back to webcam capture for testing.")
        cap = cv2.VideoCapture(0)
        saved = 0
        while saved < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            fname = output_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
            time.sleep(interval)
        cap.release()
        print(f"Done. {saved} frames saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect frames from QLabs QCar")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=300, help="Number of frames")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="Seconds between frames")
    args = parser.parse_args()

    collect_from_qlab(Path(args.output), args.num_frames, args.interval)
