#!/usr/bin/env python3
"""Panoptic demo that fuses EfficientViT (background) and YOLO (foreground).

The script mirrors the hybrid workflow prototyped inside ``test.ipynb``:

* EfficientViT-Seg predicts a dense semantic map. We keep a small subset of
  "stuff" classes (road, sky, building by default) and convert them to
  contour-based objects.
* YOLO segmentation adds "thing" instances. Each mask is resized to the input
  image size before extracting contours.

The collected objects (``name``, ``index``, ``contour``, ``box``) are exported
as JSON, and a contour visualisation is written to ``./test_results``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from eval_efficientvit_seg_model_copy import (
    CityscapesDataset,
    Resize,
    ToTensor,
    get_canvas,
)
from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from efficientvit.models.utils import resize as efficientvit_resize


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panoptic demo using EfficientViT + YOLO")
    parser.add_argument("image", type=Path, help="Path to the input RGB image")
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path("yolo11s-seg.pt"),
        help="YOLO segmentation weight file (default: yolo11s-seg.pt alongside this script)",
    )
    parser.add_argument(
        "--efficientvit-model",
        default="efficientvit-seg-b0-cityscapes",
        help="EfficientViT segmentation checkpoint name",
    )
    parser.add_argument(
        "--background-classes",
        default="0,1,2,3,4,5,8,9,10",
        help="Comma separated class IDs (Cityscapes IDs) to keep as stuff",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.01,
        help="Minimum area ratio for background masks to be accepted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Directory for JSON + visualisation outputs",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Main device used for EfficientViT (YOLO runs on CPU by default)",
    )
    parser.add_argument(
        "--save-background-canvas",
        action="store_true",
        help="Also save the pure EfficientViT segmentation canvas",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def preprocess_for_efficientvit(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Resize + normalise the image for EfficientViT and move it to *device*."""
    transform = transforms.Compose(
        [
            Resize((1024, 2048)),
            ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    feed_dict = {"data": image, "label": np.ones_like(image)}
    tensor = transform(feed_dict)["data"]
    return tensor.unsqueeze(0).to(device)


def get_contours_from_mask(binary_mask: np.ndarray) -> list[list[int]]:
    """Extract the largest contour (if any) from a binary mask."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour.reshape(-1, 2).astype(int).tolist()


def draw_objects(base_image: np.ndarray, objects: Iterable[dict], class_colors: dict) -> np.ndarray:
    vis = base_image.copy()
    for obj in objects:
        pts = np.array(obj["contour"], dtype=np.int32)
        if pts.size == 0:
            continue
        pts = pts.reshape((-1, 1, 2))
        # color = (255, 0, 0) if obj["index"] < 100 else (0, 255, 0)
        color = class_colors[obj["index"]] if obj["index"] < 100 else (0, 255, 0)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=3)
        box = obj.get("box", [0, 0, 0, 0])
        x, y = int(box[0]), int(box[1])
        cv2.putText(
            vis,
            obj["name"],
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return vis


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    original_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise RuntimeError(f"Failed to load image with OpenCV: {args.image}")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_rgb.shape[:2]

    # EfficientViT background inference -------------------------------------------------
    eff_model = create_efficientvit_seg_model(args.efficientvit_model, pretrained=True)
    eff_model.to(device)
    eff_model.eval()

    with torch.inference_mode():
        input_tensor = preprocess_for_efficientvit(original_rgb, device)
        eff_output = eff_model(input_tensor)
        if eff_output.shape[-2:] != (orig_h, orig_w):
            eff_output = efficientvit_resize(eff_output, size=(orig_h, orig_w))
        semantic_map = eff_output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    class_colors = CityscapesDataset.class_colors
    canvas = get_canvas(original_rgb, semantic_map, class_colors)
    if args.save_background_canvas:
        canvas_path = output_dir / f"{args.image.stem}_efficientvit.png"
        cv2.imwrite(str(canvas_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    background_ids = [int(x) for x in args.background_classes.split(",") if x.strip()]
    background_objects = []
    for cls_id in background_ids:
        mask = (semantic_map == cls_id).astype(np.uint8) * 255
        if mask.sum() <= args.min_area * mask.size * 255:
            continue
        contour = get_contours_from_mask(mask)
        if not contour:
            continue
        x, y, w, h = cv2.boundingRect(np.array(contour))
        background_objects.append(
            {
                "name": CityscapesDataset.classes[cls_id],
                "index": cls_id,
                "contour": contour,
                "box": [int(x), int(y), int(w), int(h)],
                "source": "efficientvit",
            }
        )

    # YOLO foreground inference ----------------------------------------------------------
    yolo_model = YOLO(str(args.yolo_weights))
    yolo_results = yolo_model(str(args.image), verbose=False)[0]
    foreground_objects = []
    if yolo_results.masks is not None:
        masks = yolo_results.masks.data.cpu().numpy()
        boxes_xywh = yolo_results.boxes.xywh.cpu().numpy()
        class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
        for idx, (mask_np, box, cls_id) in enumerate(zip(masks, boxes_xywh, class_ids)):
            mask_resized = cv2.resize(mask_np, (orig_w, orig_h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            contour = get_contours_from_mask(binary_mask)
            if not contour:
                continue
            x_b, y_b, w_b, h_b = cv2.boundingRect(np.array(contour))
            foreground_objects.append(
                {
                    "name": yolo_model.names[int(cls_id)],
                    "index": 100 + idx,
                    "contour": contour,
                    "box": [int(x_b), int(y_b), int(w_b), int(h_b)],
                    "source": "yolo",
                }
            )

    all_objects = background_objects + foreground_objects

    # Persist outputs -------------------------------------------------------------------
    json_path = output_dir / f"{args.image.stem}_objects.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_objects, f, indent=2)

    overlay = draw_objects(original_bgr, all_objects, class_colors)
    overlay_path = output_dir / f"{args.image.stem}_panoptic_contours.png"
    cv2.imwrite(str(overlay_path), overlay)

    print(f"Wrote {json_path}")
    print(f"Wrote {overlay_path}")
    if args.save_background_canvas:
        print(f"Wrote {canvas_path}")


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
