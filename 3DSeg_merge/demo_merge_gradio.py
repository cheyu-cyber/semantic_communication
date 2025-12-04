#!/usr/bin/env python3
"""Gradio app wrapper around demo_merge with an interactive 3D view."""

from __future__ import annotations

import gc
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch

from demo_merge import (
    compute_mean_depth,
    draw_objects,
    get_contours_from_mask,
    preprocess_for_efficientvit,
    resolve_device,
    run_stereo,
)
from eval_efficientvit_seg_model_copy import CityscapesDataset, get_canvas
from efficientvit.models.utils import resize as efficientvit_resize
from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from ultralytics import YOLO


DEFAULT_STEREO_CHECKPOINT = Path(__file__).resolve().parent / "models" / "generalization.ckpt"
DEFAULT_YOLO_WEIGHTS = Path("yolo11s-seg.pt")
DEFAULT_BACKGROUND_CLASSES = "2,1,3,4,5,8,9,10"


def _ensure_weights(path: Path) -> Path:
    if path.exists():
        return path
    return path


def _hard_reset_cuda() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    cudart = getattr(torch.cuda, "cudart", None)
    if callable(cudart):
        try:
            cudart().cudaDeviceReset()
        except Exception:
            pass
    gc.collect()


def _release_cuda_resources(*modules) -> None:
    if not torch.cuda.is_available():
        return
    for module in modules:
        if module is None:
            continue
        to_fn = getattr(module, "to", None)
        if callable(to_fn):
            try:
                to_fn("cpu")
            except Exception:
                pass
        model_attr = getattr(module, "model", None)
        if model_attr is not None and callable(getattr(model_attr, "to", None)):
            try:
                model_attr.to("cpu")
            except Exception:
                pass
    _hard_reset_cuda()
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if callable(ipc_collect):
        try:
            ipc_collect()
        except Exception:
            pass
    gc.collect()


def _build_point_cloud(depth_map: np.ndarray, image: np.ndarray, stride: int = 1) -> go.Figure:
    valid_mask = np.isfinite(depth_map)
    if not np.any(valid_mask):
        fig = go.Figure()
        fig.update_layout(
            scene=dict(
                xaxis_title="X (px)",
                yaxis_title="Y (px)",
                zaxis_title="Disparity",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=600,
        )
        return fig

    ys, xs = np.where(valid_mask)
    if stride > 1:
        ys = ys[::stride]
        xs = xs[::stride]

    depths = depth_map[ys, xs]
    colors_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[ys, xs]
    colors_hex = ["#%02x%02x%02x" % tuple(pixel) for pixel in colors_rgb]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=depths,
                mode="markers",
                marker=dict(size=2, color=colors_hex, opacity=0.85),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X (px)",
            yaxis_title="Y (px)",
            zaxis_title="Disparity",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
    )
    return fig


def _prepare_args(left_path: Path, right_path: Path, fp16: bool, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        left=left_path,
        right=right_path,
        stereo_checkpoint=_ensure_weights(DEFAULT_STEREO_CHECKPOINT),
        stereo_model="Fast_ACVNet_plus",
        maxdisp=192,
        yolo_weights=_ensure_weights(DEFAULT_YOLO_WEIGHTS),
        efficientvit_model="efficientvit-seg-b0-cityscapes",
        background_classes=DEFAULT_BACKGROUND_CLASSES,
        min_area=0.01,
        output_dir=output_dir,
        device="auto",
        attention_weights_only=False,
        save_background_canvas=False,
        fp16=fp16,
    )


def _run_pipeline(left_path: str, right_path: str, fp16: bool) -> Tuple[np.ndarray, np.ndarray, list[dict], go.Figure]:
    if not left_path or not right_path:
        raise gr.Error("Left and right images are required.")

    temp_dir = Path(tempfile.mkdtemp(prefix="demo_merge_gradio_"))
    args = _prepare_args(Path(left_path), Path(right_path), fp16, temp_dir)
    device = resolve_device(args.device)

    eff_model = None
    yolo_model = None
    try:
        if device.type == "cuda":
            _hard_reset_cuda()

        depth_map, raw_path, color_path = run_stereo(args.left, args.right, args, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        original_bgr = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
        if original_bgr is None:
            raise gr.Error("Failed to load the left image with OpenCV.")
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = original_rgb.shape[:2]

        eff_model = create_efficientvit_seg_model(args.efficientvit_model, pretrained=True)
        eff_model.to(device)
        eff_model.eval()
        eff_fp16_active = args.fp16 and device.type == "cuda"
        if eff_fp16_active:
            eff_model = eff_model.half()

        input_tensor = preprocess_for_efficientvit(original_rgb, device)
        if eff_fp16_active:
            input_tensor = input_tensor.half()

        try:
            with torch.inference_mode():
                eff_output = eff_model(input_tensor)
        except RuntimeError:
            if eff_fp16_active:
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
        background_ids = [int(x) for x in args.background_classes.split(",") if x.strip()]
        background_objects: list[dict] = []
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
            half=args.fp16 and device.type == "cuda",
        )[0]

        foreground_objects: list[dict] = []
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

        overlay_bgr = draw_objects(original_bgr, all_objects, class_colors)
        disparity_color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if disparity_color_bgr is None:
            raise gr.Error("Failed to load the disparity colour map.")

        silhouette_depth = np.full(depth_map.shape, np.nan, dtype=np.float32)
        silhouette_color = np.zeros_like(original_bgr)
        for obj in all_objects:
            mean_depth = obj.get("mean_depth")
            if mean_depth is None or not np.isfinite(mean_depth):
                continue
            contour = obj.get("contour", [])
            if not contour:
                continue
            pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            mask = np.zeros(depth_map.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            silhouette_depth[mask == 1] = float(mean_depth)
            silhouette_color[mask == 1] = overlay_bgr[mask == 1]

        point_cloud_fig = _build_point_cloud(silhouette_depth, silhouette_color, stride=1)
        json_payload = all_objects

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        disparity_rgb = cv2.cvtColor(disparity_color_bgr, cv2.COLOR_BGR2RGB)

        return overlay_rgb, disparity_rgb, json_payload, point_cloud_fig
    finally:
        _release_cuda_resources(eff_model, yolo_model)
        shutil.rmtree(temp_dir, ignore_errors=True)
        if device.type == "cuda":
            _hard_reset_cuda()
        gc.collect()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="3D Semantics Demo") as demo:
        gr.Markdown("## Panoptic + Depth Demo with 3D View")
        with gr.Row():
            left_input = gr.Image(label="Left Image", type="filepath")
            right_input = gr.Image(label="Right Image", type="filepath")
        fp16_checkbox = gr.Checkbox(label="Enable FP16 (where supported)", value=False)
        run_button = gr.Button("Run Pipeline")

        with gr.Row():
            overlay_output = gr.Image(label="Panoptic Overlay")
            disparity_output = gr.Image(label="Disparity Colour Map")
        json_output = gr.JSON(label="Objects Metadata")
        plot_output = gr.Plot(label="3D View")

        run_button.click(
            fn=_run_pipeline,
            inputs=[left_input, right_input, fp16_checkbox],
            outputs=[overlay_output, disparity_output, json_output, plot_output],
        )
    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
