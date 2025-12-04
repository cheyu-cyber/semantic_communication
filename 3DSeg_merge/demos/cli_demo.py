"""Command-line demo for the panoptic pipeline.

Usage:
    python demos/cli_demo.py --image path/to/image.jpg --out overlay.png

Replace `build_mock_pipeline` with wrappers for real YOLO11-seg and EfficientViT-Seg models
once available. If a disparity map is provided, the pipeline will also compute per-instance
depth and yaw estimates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from panoptic_pipeline import PanopticPipeline, build_mock_pipeline


def _draw_overlay(image: np.ndarray, pipeline: PanopticPipeline) -> np.ndarray:
    prediction = pipeline.run(image)
    overlay = image.copy()

    for idx, inst in enumerate(prediction.instances):
        color = tuple(int(c) for c in np.random.randint(64, 255, size=3))
        mask = inst.mask.astype(bool)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)
        if inst.contour is not None:
            cv2.polylines(overlay, [inst.contour], isClosed=True, color=color, thickness=2)
        if inst.bbox is not None:
            x1, y1, x2, y2 = inst.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        label = f"{inst.class_name}"
        cv2.putText(overlay, label, (inst.bbox[0], inst.bbox[1] - 4 if inst.bbox else 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panoptic segmentation demo")
    parser.add_argument("--image", required=True, help="Path to an RGB image")
    parser.add_argument("--disparity", help="Optional disparity map aligned to the image")
    parser.add_argument("--out", help="Optional path to save the overlay")
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image at {path}")
    return image


def load_disparity(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp is None:
        raise FileNotFoundError(f"Failed to load disparity at {path}")
    if disp.ndim == 3:
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
    return disp


def main() -> None:
    args = parse_args()
    image = load_image(args.image)
    disparity = load_disparity(args.disparity)

    pipeline = build_mock_pipeline()
    prediction = pipeline.run(image, disparity=disparity)
    overlay = _draw_overlay(image, pipeline)

    print("Instances:")
    for inst in prediction.instances:
        print(
            f"- {inst.class_name}: score={inst.score:.2f}, area={inst.area():.0f}, "
            f"depth={inst.depth_m if inst.depth_m is not None else 'n/a'}, "
            f"yaw={inst.yaw_deg if inst.yaw_deg is not None else 'n/a'}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, overlay)
        print(f"Saved overlay to {args.out}")


if __name__ == "__main__":
    main()
