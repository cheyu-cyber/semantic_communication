#!/usr/bin/env python3
"""Minimal inference demo for Fast-ACVNet.

This mirrors the logic in ``save_disp.py`` but processes a single stereo
pair supplied on the command line and stores the resulting disparity maps
under ``test_results``.
"""

import argparse
import os
from pathlib import Path
from collections import OrderedDict
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from datasets.data_io import get_transform
from models import __models__
from utils.experiment import make_nograd_func


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fast-ACVNet on a single stereo pair and save the disparity maps.",
    )
    parser.add_argument("--left", type=Path, help="Path to the left image.")
    parser.add_argument("--right", type=Path, help="Path to the right image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a pretrained Fast-ACVNet checkpoint (same as used by save_disp.py).",
    )
    parser.add_argument(
        "--model",
        default="Fast_ACVNet_plus",
        choices=__models__.keys(),
        help="Model variant to instantiate.",
    )
    parser.add_argument("--maxdisp", type=int, default=192, help="Maximum disparity for the network.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Directory where disparity outputs will be written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Computation device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--attention-weights-only",
        action="store_true",
        help="Instantiate the network in attention-weights-only mode (matches save_disp flag).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Batch size for inference; keep at 1 for single-image demo.",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _prepare_tensor(image: Image.Image) -> tuple[np.ndarray, int, int]:
    """Apply the same preprocessing and padding strategy as save_disp."""
    processed = get_transform()(image).numpy()
    _, h, w = processed.shape

    # Pad to multiples of 32 on the top/right edges (mirrors dataset preprocessing).
    pad_h = ((h + 31) // 32) * 32 - h
    pad_w = ((w + 31) // 32) * 32 - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Computed negative padding – unexpected input size.")

    padded = np.pad(
        processed,
        ((0, 0), (pad_h, 0), (0, pad_w)),
        mode="constant",
        constant_values=0,
    )
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
def _infer(model: nn.Module, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    model.eval()
    outputs = model(left, right)
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def _save_disparity_maps(raw_disp: np.ndarray, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{stem}_disp.png"
    color_path = output_dir / f"{stem}_disp_color.png"

    disp_uint = np.round(raw_disp * 256.0).astype(np.uint16)
    cv2.imwrite(str(raw_path), disp_uint)
    disp_norm = cv2.normalize(raw_disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(color_path), disp_color)
    return raw_path, color_path


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)

    left_img = _load_image(args.left)
    right_img = _load_image(args.right)

    left_np, top_pad, right_pad = _prepare_tensor(left_img)
    right_np, right_top_pad, right_right_pad = _prepare_tensor(right_img)
    if (top_pad, right_pad) != (right_top_pad, right_right_pad):
        raise ValueError("Left and right images produced mismatched padding sizes. Check image alignment.")

    left_tensor = torch.from_numpy(left_np).unsqueeze(0).to(device)
    right_tensor = torch.from_numpy(right_np).unsqueeze(0).to(device)

    model = __models__[args.model](args.maxdisp, args.attention_weights_only)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda().half()
    model.to(device)

    start = time.perf_counter()
    _load_checkpoint(model, args.checkpoint, device)
    print(f"Checkpoint load time: {time.perf_counter() - start:.03f}s")
    start = time.perf_counter()
    disp = _infer(model, left_tensor, right_tensor)
    disp = disp.squeeze(0).squeeze(0).detach().cpu().numpy()
    if top_pad > 0:
        disp = disp[top_pad:]
    if right_pad > 0:
        disp = disp[:, :-right_pad]
    print(f"Inference time: {time.perf_counter() - start:.03f}s")
    raw_path, color_path = _save_disparity_maps(disp.astype(np.float32), args.output_dir, args.left.stem)

    print(f"Saved raw disparity to: {raw_path}")
    print(f"Saved colorized disparity to: {color_path}")


if __name__ == "__main__":
    main()
