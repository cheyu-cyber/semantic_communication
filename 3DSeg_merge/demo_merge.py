#!/usr/bin/env python3
"""Panoptic + depth demo combining Fast-ACVNet, EfficientViT, and YOLO.

Pipeline order:
1. Fast-ACVNet computes a disparity (pseudo depth) map from a stereo pair.
2. EfficientViT-Seg provides background "stuff" regions.
3. YOLO segmentation contributes foreground "thing" instances.

For every reported object we export:
- name
- index (class or shifted class id)
- numerator (running instance number)
- contour (list of [x, y] pairs)
- box (x, y, w, h)
- mean_depth (mean disparity value within the contour mask)

Outputs are written under ``--output-dir``:
- Disparity PNG (16-bit) and colour visualisation
- JSON metadata for detected objects
- Contour overlay image (and optional EfficientViT canvas)

Set ``--fp16`` to enable half precision on CUDA for Fast-ACVNet, EfficientViT, and YOLO stages.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from datasets.data_io import get_transform
from eval_efficientvit_seg_model_copy import (
    CityscapesDataset,
    Resize,
    ToTensor,
    get_canvas,
)
from efficientvit.models.utils import resize as efficientvit_resize
from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from models import __models__
from utils.experiment import make_nograd_func


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Fast-ACVNet depth with EfficientViT+YOLO panoptic outputs",
    )
    parser.add_argument("left", type=Path, help="Path to the left RGB image")
    parser.add_argument("right", type=Path, help="Path to the right RGB image")
    parser.add_argument(
        "--stereo-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint for Fast-ACVNet (same format as save_disp.py)",
    )
    parser.add_argument(
        "--stereo-model",
        default="Fast_ACVNet_plus",
        choices=__models__.keys(),
        help="Fast-ACVNet model variant",
    )
    parser.add_argument(
        "--maxdisp",
        type=int,
        default=192,
        help="Maximum disparity used by Fast-ACVNet",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path("yolo11s-seg.pt"),
        help="YOLO segmentation weights (default: yolo11s-seg.pt)",
    )
    parser.add_argument(
        "--efficientvit-model",
        default="efficientvit-seg-b0-cityscapes",
        help="EfficientViT segmentation model name",
    )
    parser.add_argument(
        "--background-classes",
        default="1,2,3,4,5,8,9,10",
        help="Comma separated Cityscapes IDs treated as background stuff",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.01,
        help="Minimum area ratio for background masks to keep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Directory for disparity, overlays, and JSON",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Primary computation device",
    )
    parser.add_argument(
        "--attention-weights-only",
        action="store_true",
        help="Instantiate Fast-ACVNet in attention-weights-only mode",
    )
    parser.add_argument(
        "--save-background-canvas",
        action="store_true",
        help="Also save the EfficientViT-only segmentation canvas",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 inference on CUDA stages where supported",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def get_contours_from_mask(binary_mask: np.ndarray) -> list[list[int]]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour.reshape(-1, 2).astype(int).tolist()


def draw_objects(base_image: np.ndarray, objects: Iterable[dict], class_colors: tuple) -> np.ndarray:
    vis = base_image.copy()
    for obj in objects:
        pts = np.array(obj["contour"], dtype=np.int32)
        if pts.size == 0:
            continue
        pts = pts.reshape((-1, 1, 2))
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


def preprocess_for_efficientvit(image: np.ndarray, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            Resize((1024, 2048)),
            ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform({"data": image, "label": np.ones_like(image)})["data"]
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Fast-ACVNet helpers
# ---------------------------------------------------------------------------

def _prepare_tensor(image: Image.Image) -> tuple[np.ndarray, int, int]:
    processed = get_transform()(image).numpy()
    _, h, w = processed.shape
    pad_h = ((h + 31) // 32) * 32 - h
    pad_w = ((w + 31) // 32) * 32 - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Computed negative padding – unexpected input size.")
    padded = np.pad(processed, ((0, 0), (pad_h, 0), (0, pad_w)), mode="constant", constant_values=0)
    return padded, pad_h, pad_w


def _load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model", state)
    if isinstance(model, nn.DataParallel):
        model.load_state_dict(state_dict)
        return
    if all(k.startswith("module.") for k in state_dict.keys()):
        stripped = OrderedDict((k[len("module."):], v) for k, v in state_dict.items())
        model.load_state_dict(stripped)
    else:
        model.load_state_dict(state_dict)


@make_nograd_func
def _infer_stereo(model: nn.Module, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    model.eval()
    outputs = model(left, right)
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def run_stereo(left_path: Path, right_path: Path, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, Path, Path]:
    use_fp16 = args.fp16 and device.type == "cuda"
    left_img = Image.open(left_path).convert("RGB")
    right_img = Image.open(right_path).convert("RGB")

    left_np, top_pad, right_pad = _prepare_tensor(left_img)
    right_np, top_pad_r, right_pad_r = _prepare_tensor(right_img)
    if (top_pad, right_pad) != (top_pad_r, right_pad_r):
        raise ValueError("Left and right images produced mismatched padding sizes.")

    tensor_dtype = torch.float16 if use_fp16 else torch.float32
    left_tensor = torch.from_numpy(left_np).unsqueeze(0).to(device=device, dtype=tensor_dtype)
    right_tensor = torch.from_numpy(right_np).unsqueeze(0).to(device=device, dtype=tensor_dtype)

    model = __models__[args.stereo_model](args.maxdisp, args.attention_weights_only)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    tic = time.perf_counter()
    _load_checkpoint(model, args.stereo_checkpoint, device)
    load_time = time.perf_counter() - tic

    if use_fp16:
        model = model.half()

    tic = time.perf_counter()
    try:
        disp = _infer_stereo(model, left_tensor, right_tensor)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if use_fp16 and "grid_sampler" in msg:
            print("Fast-ACVNet FP16 not supported (grid sampler); retrying with FP32.")
            model = model.float()
            left_tensor = left_tensor.float()
            right_tensor = right_tensor.float()
            use_fp16 = False
            disp = _infer_stereo(model, left_tensor, right_tensor)
        else:
            raise
    infer_time = time.perf_counter() - tic

    disp = disp.float().squeeze(0).squeeze(0).detach().cpu().numpy()
    if top_pad > 0:
        disp = disp[top_pad:]
    if right_pad > 0:
        disp = disp[:, :-right_pad]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{left_path.stem}_disp.png"
    color_path = output_dir / f"{left_path.stem}_disp_color.png"
    disp_uint = np.round(disp * 256.0).astype(np.uint16)
    cv2.imwrite(str(raw_path), disp_uint)
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(color_path), disp_color)

    print(f"Stereo checkpoint load time: {load_time:.03f}s")
    print(f"Stereo inference time: {infer_time:.03f}s")
    print(f"Saved disparity to {raw_path}")
    print(f"Saved coloured disparity to {color_path}")

    result = disp.astype(np.float32)

    if device.type == "cuda":
        model.to("cpu")
    del model
    del left_tensor
    del right_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, raw_path, color_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_mean_depth(depth_map: np.ndarray, mask: np.ndarray) -> float:
    flat_mask = mask.astype(bool)
    if not flat_mask.any():
        return float("nan")
    return float(depth_map[flat_mask].mean())


def run_pipeline(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and not use_fp16:
        print("FP16 requested but CUDA device unavailable; continuing with FP32.")
    depth_map, _, _ = run_stereo(args.left, args.right, args, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    original_bgr = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise RuntimeError(f"Failed to load image with OpenCV: {args.left}")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_rgb.shape[:2]

    # EfficientViT background inference
    eff_model = create_efficientvit_seg_model(args.efficientvit_model, pretrained=True)
    eff_model.to(device)
    eff_model.eval()
    eff_fp16_active = use_fp16
    if eff_fp16_active:
        eff_model = eff_model.half()

    input_tensor = preprocess_for_efficientvit(original_rgb, device)
    if eff_fp16_active:
        input_tensor = input_tensor.half()

    try:
        with torch.inference_mode():
            eff_output = eff_model(input_tensor)
    except RuntimeError as exc:
        if eff_fp16_active:
            print("EfficientViT FP16 failed; retrying with FP32.")
            eff_fp16_active = False
            eff_model = eff_model.float()
            input_tensor = input_tensor.float()
            with torch.inference_mode():
                eff_output = eff_model(input_tensor)
        else:
            raise

    eff_output = eff_output.float()
    if eff_output.shape[-2:] != (orig_h, orig_w):
        eff_output = efficientvit_resize(eff_output, size=(orig_h, orig_w))
    semantic_map = eff_output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    class_colors = CityscapesDataset.class_colors
    canvas = get_canvas(original_rgb, semantic_map, class_colors)
    if args.save_background_canvas:
        canvas_path = args.output_dir / f"{args.left.stem}_efficientvit.png"
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
        mean_depth = compute_mean_depth(depth_map, mask)
        background_objects.append(
            {
                "name": CityscapesDataset.classes[cls_id],
                "index": cls_id,
                "contour": contour,
                "box": [int(x), int(y), int(w), int(h)],
                "mean_depth": mean_depth,
                "source": "efficientvit",
            }
        )

    # YOLO foreground inference
    yolo_model = YOLO(str(args.yolo_weights))
    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        device_str = f"cuda:{device_index}"
    else:
        device_str = "cpu"
    yolo_results = yolo_model.predict(
        source=str(args.left),
        verbose=False,
        device=device_str,
        half=use_fp16,
    )[0]
    foreground_objects = []
    if yolo_results.masks is not None:
        masks = yolo_results.masks.data.cpu().numpy()
        class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
        for idx, (mask_np, cls_id) in enumerate(zip(masks, class_ids)):
            mask_resized = cv2.resize(mask_np, (orig_w, orig_h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            contour = get_contours_from_mask(binary_mask)
            if not contour:
                continue
            x_b, y_b, w_b, h_b = cv2.boundingRect(np.array(contour))
            mean_depth = compute_mean_depth(depth_map, binary_mask)
            foreground_objects.append(
                {
                    "name": yolo_model.names[int(cls_id)],
                    "index": 100 + idx,
                    "contour": contour,
                    "box": [int(x_b), int(y_b), int(w_b), int(h_b)],
                    "mean_depth": mean_depth,
                    "source": "yolo",
                }
            )

    all_objects = background_objects + foreground_objects
    for numerator, entry in enumerate(all_objects, start=1):
        entry["numerator"] = numerator

    json_path = args.output_dir / f"{args.left.stem}_objects.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_objects, f, indent=2)

    overlay = draw_objects(original_bgr, all_objects, class_colors)
    overlay_path = args.output_dir / f"{args.left.stem}_panoptic_contours.png"
    cv2.imwrite(str(overlay_path), overlay)

    print(f"Wrote {json_path}")
    print(f"Wrote {overlay_path}")
    if args.save_background_canvas:
        print(f"Wrote {canvas_path}")


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
