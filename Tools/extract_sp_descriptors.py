#!/usr/bin/env python3
"""
Extract SuperPoint descriptors from dataset images for BoW vocabulary training.

v3: accuracy-focused — adaptive threshold, spatial grid forcing, quality filter.

Pipeline per image:
  1. ONNX inference -> semi + desc
  2. softmax + reshape -> heatmap
  3. NMS -> candidate keypoints + scores
  4. Adaptive confidence: percentile-based threshold (not fixed)
  5. Spatial grid forcing: ensure even coverage across image regions
  6. Score-ranked Top-K truncation
  7. Bilinear interpolation -> sub-pixel descriptors
  8. Quality filter: remove degenerate descriptors (low norm / high NaN)
  9. L2 normalize

Output binary format (compatible with train_sp_vocabulary C++ tool):
  - num_images  (int32)
  - per-image keypoint counts  (int32 × num_images)
  - descriptors  (float32 × N × 256, L2-normalized)

Usage:
  conda activate superpoint
  python3 Tools/extract_sp_descriptors.py <image_dir> -o <output.bin>
"""

import os
import sys
import glob
import struct
import argparse
import numpy as np
import cv2
import onnxruntime as ort


# ──────────────────────────────────────────────
# Core: softmax + reshape + NMS
# ──────────────────────────────────────────────

def softmax_64ch(semi):
    """Dustbin-free softmax + reshape 64ch -> 8x8 grid -> full-resolution heatmap."""
    cells = semi[:64]
    cells = cells - cells.max(axis=0, keepdims=True)
    exp_cells = np.exp(cells)
    prob = exp_cells / exp_cells.sum(axis=0, keepdims=True)

    featH, featW = prob.shape[1], prob.shape[2]
    heatmap = np.zeros((featH * 8, featW * 8), dtype=np.float32)
    for dy in range(8):
        for dx in range(8):
            heatmap[dy::8, dx::8] = prob[dy * 8 + dx]
    return heatmap


def nms_dilate(scores, nms_radius):
    """Fast NMS via cv2.dilate. Returns ALL local maxima above 0."""
    kernel_size = 2 * nms_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    local_max = cv2.dilate(scores, kernel)
    is_max = (scores == local_max) & (scores > 0)
    ys, xs = np.where(is_max)
    return ys, xs, scores[ys, xs]


# ──────────────────────────────────────────────
# v3 新增: 空间网格均匀采样
# ──────────────────────────────────────────────

def spatial_grid_select(ys, xs, scores, H, W, grid_n=4, max_per_cell=None):
    """
    Divide image into grid_n×grid_n cells, take top points from each cell.
    This prevents keypoints from clustering in one textured region and ensures
    the vocabulary sees features from all parts of the image.

    Args:
        ys, xs, scores:  candidate keypoints (pre-sorted by score descending)
        H, W:           image dimensions
        grid_n:         number of cells per axis (4×4 = 16 cells)
        max_per_cell:   max points per cell (None = no limit)

    Returns:
        mask: boolean array, True for selected keypoints
    """
    cell_h, cell_w = H / grid_n, W / grid_n

    # Assign each keypoint to a grid cell
    cy = np.clip((ys / cell_h).astype(int), 0, grid_n - 1)
    cx = np.clip((xs / cell_w).astype(int), 0, grid_n - 1)
    cell_id = cy * grid_n + cx  # unique cell index

    # For each cell, keep top-scoring points
    mask = np.zeros(len(ys), dtype=bool)
    for c in range(grid_n * grid_n):
        cell_mask = cell_id == c
        indices = np.where(cell_mask)[0]
        if len(indices) == 0:
            continue
        # Points are already sorted by score (caller ensures this)
        if max_per_cell is not None and len(indices) > max_per_cell:
            indices = indices[:max_per_cell]
        mask[indices] = True

    return mask


# ──────────────────────────────────────────────
# Core extraction pipeline
# ──────────────────────────────────────────────

def extract_descriptors(session, img_float, conf_percentile, nms_radius,
                        max_kp, grid_n=4, max_per_cell=None):
    """
    SuperPoint inference with adaptive threshold + spatial grid + quality filter.

    Args:
        session:          ONNX Runtime session
        img_float:        (H, W) float32 grayscale [0,1]
        conf_percentile:  keep top N% by confidence (e.g. 0.3 = top 30%)
        nms_radius:       NMS suppression radius
        max_kp:           absolute max keypoints per image
        grid_n:           spatial grid cells per axis (4 = 4x4 = 16 cells)
        max_per_cell:     max keypoints per grid cell (None = no limit)

    Returns:
        descriptors: (N, 256) float32, L2-normalized
    """
    H, W = img_float.shape
    input_tensor = img_float[np.newaxis, np.newaxis, :, :].astype(np.float32)
    semi, desc = session.run(None, {'input': input_tensor})

    semi = semi[0]  # (65, featH, featW)
    desc = desc[0]  # (256, featH, featW)
    featH, featW = desc.shape[1], desc.shape[2]

    # Softmax + reshape -> heatmap
    heatmap = softmax_64ch(semi)

    # NMS -> all local maxima
    ys, xs, scores = nms_dilate(heatmap, nms_radius)
    if len(ys) == 0:
        return np.zeros((0, 256), dtype=np.float32)

    # v3 改动 1: 自适应置信度 — 用百分位数代替固定阈值
    # 保留置信度最高的前 conf_percentile 的关键点
    thresh = np.percentile(scores, (1.0 - conf_percentile) * 100)
    thresh = max(thresh, 1e-4)  # floor to avoid degenerate images
    mask = scores >= thresh
    if not np.any(mask):
        return np.zeros((0, 256), dtype=np.float32)
    ys, xs, scores = ys[mask], xs[mask], scores[mask]

    # Sort by score descending
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    # v3 改动 2: 空间网格均匀采样
    if grid_n > 0:
        grid_mask = spatial_grid_select(ys, xs, scores, H, W,
                                        grid_n=grid_n,
                                        max_per_cell=max_per_cell)
        ys, xs, scores = ys[grid_mask], xs[grid_mask], scores[grid_mask]

    # Top-K truncation
    if len(ys) > max_kp:
        ys, xs, scores = ys[:max_kp], xs[:max_kp], scores[:max_kp]

    if len(ys) == 0:
        return np.zeros((0, 256), dtype=np.float32)

    # Bilinear interpolation
    fx = (xs.astype(np.float32) - 4.0 + 0.5) / (featW * 8.0 - 4.0 - 0.5) * 2.0 - 1.0
    fy = (ys.astype(np.float32) - 4.0 + 0.5) / (featH * 8.0 - 4.0 - 0.5) * 2.0 - 1.0
    fx = (fx + 1.0) * (featW - 1.0) / 2.0
    fy = (fy + 1.0) * (featH - 1.0) / 2.0
    fx = np.clip(fx, 0, featW - 1.001)
    fy = np.clip(fy, 0, featH - 1.001)

    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = np.minimum(x0 + 1, featW - 1)
    y1 = np.minimum(y0 + 1, featH - 1)

    wa = (x1.astype(np.float32) - fx) * (y1.astype(np.float32) - fy)
    wb = (fx - x0.astype(np.float32)) * (y1.astype(np.float32) - fy)
    wc = (x1.astype(np.float32) - fx) * (fy - y0.astype(np.float32))
    wd = (fx - x0.astype(np.float32)) * (fy - y0.astype(np.float32))

    descriptors = (desc[:, y0, x0] * wa[np.newaxis, :] +
                   desc[:, y0, x1] * wb[np.newaxis, :] +
                   desc[:, y1, x0] * wc[np.newaxis, :] +
                   desc[:, y1, x1] * wd[np.newaxis, :]).T  # (N, 256)

    # v3 改动 3: 描述子质量过滤
    # 剔除插值后范数异常低的退化描述子
    norms = np.linalg.norm(descriptors, axis=1)
    valid = norms > 0.1  # L2-normalized 后范数应接近 1.0, <0.1 说明插值出问题
    if not np.any(valid):
        return np.zeros((0, 256), dtype=np.float32)
    descriptors = descriptors[valid]

    # L2 normalize
    norms = norms[valid:valid+1] if valid.ndim == 0 else norms[valid]
    norms = np.maximum(norms.reshape(-1, 1), 1e-6)
    descriptors = descriptors / norms

    return descriptors.astype(np.float32)


# ──────────────────────────────────────────────
# argparse
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Extract SuperPoint descriptors for BoW vocabulary training (v3: accuracy)')

    p.add_argument('image_dir', type=str,
                   help='Directory containing PNG/JPG images')
    p.add_argument('-o', '--output', type=str, default='sp_descriptors.bin',
                   help='Output binary file path')
    p.add_argument('--model', type=str,
                   default='/home/yuan/ORB_SLAM3/Models/superpoint_v1.onnx',
                   help='Path to SuperPoint ONNX model')
    p.add_argument('--max_keypoints', type=int, default=300,
                   help='Max keypoints per image (default: 300)')
    p.add_argument('--skip_frames', type=int, default=5,
                   help='Process every N-th image (default: 5)')
    p.add_argument('--conf_percentile', type=float, default=0.3,
                   help='Keep top N%% keypoints by confidence (default: 0.3 = top 30%%)')
    p.add_argument('--nms_radius', type=int, default=4,
                   help='NMS suppression radius in pixels (default: 4)')
    p.add_argument('--grid_n', type=int, default=4,
                   help='Spatial grid cells per axis, 0=disable (default: 4 = 4x4)')
    p.add_argument('--max_per_cell', type=int, default=50,
                   help='Max keypoints per grid cell (default: 50)')

    return p.parse_args()


# ──────────────────────────────────────────────
# Main: skip frames + extract + binary write
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.png')))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))
    if not image_paths:
        print(f"ERROR: No images found in {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    sampled = image_paths[::args.skip_frames]

    print(f"Dataset:          {args.image_dir}")
    print(f"Total images:     {len(image_paths)}")
    print(f"Skip frames:      {args.skip_frames}  ->  sampled: {len(sampled)}")
    print(f"Max keypoints:    {args.max_keypoints}")
    print(f"Conf percentile:  {args.conf_percentile}  (top {args.conf_percentile*100:.0f}%%)")
    print(f"NMS radius:       {args.nms_radius}")
    print(f"Spatial grid:     {args.grid_n}x{args.grid_n}  ({args.grid_n**2} cells, max {args.max_per_cell}/cell)")
    print(f"Model:            {args.model}")
    print()

    print("Loading ONNX model...", flush=True)
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    print("Model loaded.", flush=True)

    all_num_kp = []
    all_descriptors = []
    total_kp = 0

    for i, img_path in enumerate(sampled):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            all_num_kp.append(0)
            continue

        H, W = img.shape
        img_float = (img.astype(np.float32) / 255.0)[:((H // 8) * 8), :((W // 8) * 8)]

        desc = extract_descriptors(session, img_float,
                                   conf_percentile=args.conf_percentile,
                                   nms_radius=args.nms_radius,
                                   max_kp=args.max_keypoints,
                                   grid_n=args.grid_n,
                                   max_per_cell=args.max_per_cell)

        n_kp = len(desc)
        all_num_kp.append(n_kp)
        if n_kp > 0:
            all_descriptors.append(desc)
            total_kp += n_kp

        print(f"  [{i+1:4d}/{len(sampled)}] {os.path.basename(img_path)}: "
              f"{n_kp:4d} kp  (global idx {i * args.skip_frames})",
              flush=True)

    print(f"\nTotal: {total_kp} keypoints from {len(sampled)} images")

    with open(args.output, 'wb') as f:
        f.write(struct.pack('<i', len(sampled)))
        for n in all_num_kp:
            f.write(struct.pack('<i', n))
        for desc in all_descriptors:
            f.write(desc.tobytes())

    file_size = os.path.getsize(args.output)
    print(f"Saved: {args.output} ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    main()
