"""Gradio demo for the panoptic pipeline using mock segmenters.

Replace `build_mock_pipeline` with real YOLO11-seg and EfficientViT-Seg wrappers when
available. The demo overlays instance masks and contours on the input image.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np

from panoptic_pipeline import PanopticPipeline, build_mock_pipeline

COLORS = [(255, 87, 51), (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182), (52, 73, 94)]


def _color_for_index(idx: int) -> Tuple[int, int, int]:
    return COLORS[idx % len(COLORS)]


def visualize_prediction(image: np.ndarray, pipeline: PanopticPipeline):
    prediction = pipeline.run(image)
    overlay = image.copy()

    for idx, inst in enumerate(prediction.instances):
        color = _color_for_index(idx)
        mask = inst.mask.astype(bool)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)
        if inst.contour is not None:
            cv2.polylines(overlay, [inst.contour], isClosed=True, color=color, thickness=2)
        if inst.bbox is not None:
            x1, y1, x2, y2 = inst.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        label = f"{inst.class_name}"
        cv2.putText(overlay, label, (inst.bbox[0], inst.bbox[1] - 4 if inst.bbox else 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return overlay, prediction


def format_instances(instances) -> List[str]:
    rows = []
    for inst in instances:
        rows.append(
            f"{inst.class_name} | score={inst.score:.2f} | area={inst.area():.0f} | "
            f"depth={inst.depth_m if inst.depth_m is not None else 'n/a'} | "
            f"yaw={inst.yaw_deg if inst.yaw_deg is not None else 'n/a'}"
        )
    return rows


def main() -> None:
    pipeline = build_mock_pipeline()

    with gr.Blocks(title="Panoptic Segmentation (YOLO11-seg + EfficientViT-Seg)") as demo:
        gr.Markdown("""# Panoptic Segmentation Demo
Upload an image to run the mock YOLO11-seg (things) + EfficientViT-Seg (stuff) pipeline.
Replace the mock segmenters with real models to obtain meaningful predictions.
""")

        with gr.Row():
            inp = gr.Image(type="numpy", label="Input image")
            out_overlay = gr.Image(type="numpy", label="Overlay")
        out_text = gr.JSON(label="Per-instance output")

        def _run(image: np.ndarray):
            overlay, prediction = visualize_prediction(image, pipeline)
            return overlay, [inst.__dict__ for inst in prediction.instances]

        inp.change(_run, inputs=inp, outputs=[out_overlay, out_text])

    demo.launch()


if __name__ == "__main__":
    main()
