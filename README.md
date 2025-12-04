<<<<<<< HEAD
# Semantic_Segmentation_Network

This repository implements a panoptic segmentation pipeline that fuses a **thing** segmenter (YOLO11-seg) with a **stuff** segmenter (EfficientViT-Seg) and augments results with depth and orientation estimates. The code includes:

- `panoptic_pipeline.py`: a lightweight, framework-agnostic pipeline with merging, contour extraction, depth, and yaw helpers.
- `demos/gradio_app.py`: a Gradio UI wired to the pipeline (mock models by default; swap in real YOLO11-seg/EfficientViT-Seg wrappers when available).
- `demos/cli_demo.py`: a CLI demo that prints per-object metadata and can save an overlay.
- `demos/panoptic_demo.ipynb`: a notebook walkthrough that builds the pipeline, runs a mock inference, and visualizes the overlay.

## Pipeline overview
1. **Input acquisition (stereo pair or monocular RGB).** Calibrate and rectify the stereo pair; keep intrinsic/extrinsic parameters for later depth and yaw estimation.
2. **Pre-processing.** Normalize and resize to each model's native resolution, preserving aspect ratio. Maintain the scaling metadata to re-project masks and boxes back to the original coordinates.
3. **Thing segmentation — YOLO11-seg.** Run YOLO11-seg to obtain per-instance masks, classes, confidences, and bounding boxes for countable objects. Apply confidence and mask-size filters; optionally run non-maximum suppression (NMS) on masks.
4. **Stuff segmentation — EfficientViT-Seg.** Run EfficientViT-Seg for background and amorphous regions. Keep the per-pixel class probabilities to support overlap resolution.
5. **Panoptic merge.**
   - Warp outputs back to the original image space using the stored scaling factors.
   - For overlapping regions, prioritize YOLO11-seg instances; fill remaining pixels with EfficientViT-Seg labels.
   - Where YOLO11-seg masks conflict, apply mask-aware NMS (e.g., based on intersection-over-union) to retain the highest-confidence instance per overlap area.
6. **Contour extraction.** For each merged instance, trace the polygon contour (e.g., via `cv2.findContours` on the binary mask). Store simplified polygons for compactness if needed.
7. **Depth estimation (recommended after panoptic merge).**
   - Use the stereo pair to compute a dense disparity map (e.g., Semi-Global Matching or a learned stereo depth network) and convert disparity to depth using calibration.
   - For each merged instance, compute a robust depth statistic (median or trimmed mean) over its mask to output a single depth value.
8. **Yaw estimation (optional).** Use 3D points recovered from the depth map within the mask and camera intrinsics to fit a local plane or principal components; derive the yaw angle relative to the image XY plane or vehicle frame.
9. **Output formatting.** Emit a record per object: `{class_name, contour (list of xy points), bbox (optional), depth_m (single value), yaw_deg (optional)}`.

## Running the demos

### Gradio UI
```bash
pip install -r requirements.txt  # Ensure opencv-python, gradio, numpy are available
python demos/gradio_app.py
```
Upload an image to visualize overlays and per-instance fields. Replace the mock builders in `gradio_app.py` with real YOLO11-seg and EfficientViT-Seg wrappers when you have the weights.

### CLI demo
```bash
python demos/cli_demo.py --image path/to/image.jpg --out overlay.png
# Optional depth/yaw if a disparity map is available:
python demos/cli_demo.py --image path/to/image.jpg --disparity path/to/disp.png --out overlay.png
```
The CLI prints per-instance metadata and writes an overlay image if `--out` is provided.

### Notebook walkthrough
Open `demos/panoptic_demo.ipynb` in Jupyter or VS Code. The notebook uses the mock models to:

- build the pipeline
- generate a synthetic image and optional disparity map
- run inference and print per-instance fields
- visualize contours and labels over the input image

## Should depth run before or after segmentation?
Placing depth **after** segmentation is preferred:
- The merged panoptic masks provide clean regions to aggregate depth, producing stable per-object depth values.
- Depth noise can be filtered by the masks, improving robustness versus pixel-wise depth first.
- The segmentation stage remains agnostic to stereo quality, simplifying deployment when stereo is unavailable (monocular fallback).

Running depth **before** segmentation is only beneficial if you need depth-aware NMS or cross-view consistency checks, but it increases complexity and requires propagating dense depth through both segmenters.

## Practical notes
- Synchronize the stereo pair and run rectification to avoid disparity artifacts at object boundaries.
- Keep model-specific preprocessing in sync with training (color space, normalization, resize strategy).
- Profile both models; consider batching or using TensorRT to meet latency targets.
- Persist calibration and scaling metadata with predictions so downstream modules (tracking, planning) can correctly interpret contours, boxes, and depth.
=======
# semantic_communication
BU MS EC601 Project
>>>>>>> 8194e9acb1c6fa9974651d70d7f9448302c2da50
