"""src/data/split_dataset.py
Split annotated data (YOLO format) into train / val / test sets.

Expected input structure (--src):
  annotated/
    images/  *.jpg / *.png
    labels/  *.txt  (YOLO format)

Output structure (--dst):
  splits/
    train/images/  train/labels/
    val/images/    val/labels/
    test/images/   test/labels/
"""

import argparse
import shutil
import random
from pathlib import Path


def split_dataset(src: Path, dst: Path,
                  train_r=0.7, val_r=0.2, test_r=0.1,
                  seed=42):
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Ratios must sum to 1"

    img_dir = src / "images"
    lbl_dir = src / "labels"

    images = sorted(img_dir.glob("*.*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split_name, split_imgs in splits.items():
        out_img = dst / split_name / "images"
        out_lbl = dst / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy2(img_path, out_img / img_path.name)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl / lbl_path.name)
            else:
                # Create empty label file for images with no signs
                (out_lbl / (img_path.stem + ".txt")).touch()

        print(f"  {split_name:5s}: {len(split_imgs):4d} images")

    print(f"\nSplit complete → {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/annotated")
    parser.add_argument("--dst", default="data/splits")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val",   type=float, default=0.2)
    parser.add_argument("--test",  type=float, default=0.1)
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    split_dataset(
        Path(args.src), Path(args.dst),
        train_r=args.train, val_r=args.val, test_r=args.test,
        seed=args.seed,
    )
