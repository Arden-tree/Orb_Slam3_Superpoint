#!/usr/bin/env python3
"""
Extract SuperPoint descriptors from dataset images for BoW vocabulary training.

Uses ONNX Runtime to run SuperPoint inference, then applies confidence thresholding,
NMS, score-ranked Top-K selection to produce high-quality descriptors.

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
# 流程二核心: NMS + 置信度过滤 + 分数排序 + Top-K 截断
# ──────────────────────────────────────────────

def softmax_64ch(semi):
    """
    Dustbin-free softmax + reshape 64 channels -> 8x8 grid -> full-resolution heatmap.

    Args:
        semi: (65, H/8, W/8) raw logits from ONNX
    Returns:
        heatmap: (H, W) full-resolution score map
    """
    cells = semi[:64]  # (64, featH, featW)

    # Softmax over 64 sub-pixel channels (exclude dustbin ch64)
    cells = cells - cells.max(axis=0, keepdims=True)
    exp_cells = np.exp(cells)
    prob = exp_cells / exp_cells.sum(axis=0, keepdims=True)

    # Reshape: 64 channels -> 8x8 grid -> full resolution (H, W)
    featH, featW = prob.shape[1], prob.shape[2]
    heatmap = np.zeros((featH * 8, featW * 8), dtype=np.float32)

    for dy in range(8):
        for dx in range(8):
            heatmap[dy::8, dx::8] = prob[dy * 8 + dx]

    return heatmap


def nms_dilate(scores, nms_radius):
    """
    Fast NMS via cv2.dilate (O(1) per pixel using max-pool).
    Returns ALL local maxima above 0, unsorted.
    """
    kernel_size = 2 * nms_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    local_max = cv2.dilate(scores, kernel)
    is_max = (scores == local_max) & (scores > 0)
    ys, xs = np.where(is_max)
    return ys, xs, scores[ys, xs]


def extract_descriptors(session, img_float, conf_thresh, nms_radius, max_kp):
    """
    SuperPoint inference pipeline with quality-gated Top-K selection.

    Pipeline:
      1. ONNX inference -> semi (65, H/8, W/8) + desc (256, H/8, W/8)
      2. softmax + reshape -> full-resolution heatmap
      3. NMS -> candidate keypoints + scores
      4. Confidence threshold -> filter weak points
      5. Sort by score descending -> rank quality
      6. Top-K truncation -> keep only the elite
      7. Bilinear interpolation -> sub-pixel descriptors
      8. L2 normalize

    Args:
        session:      ONNX Runtime session
        img_float:    (H, W) float32 grayscale [0,1], padded to multiple of 8
        conf_thresh:  minimum keypoint confidence
        nms_radius:   NMS suppression radius in pixels
        max_kp:       maximum keypoints to keep per image

    Returns:
        descriptors: (N, 256) float32, L2-normalized, score-ranked
    """
    input_tensor = img_float[np.newaxis, np.newaxis, :, :].astype(np.float32)
    semi, desc = session.run(None, {'input': input_tensor})

    semi = semi[0]  # (65, featH, featW)
    desc = desc[0]  # (256, featH, featW)
    featH, featW = desc.shape[1], desc.shape[2]

    # Step 2: softmax + reshape -> heatmap
    heatmap = softmax_64ch(semi)

    # Step 3: NMS -> all local maxima
    ys, xs, scores = nms_dilate(heatmap, nms_radius)

    if len(ys) == 0:
        return np.zeros((0, 256), dtype=np.float32)

    # Step 4: confidence threshold (cheap, do BEFORE expensive interpolation)
    mask = scores >= conf_thresh
    if not np.any(mask):
        return np.zeros((0, 256), dtype=np.float32)
    ys, xs, scores = ys[mask], xs[mask], scores[mask]

    # Step 5: sort by score descending
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    # Step 6: Top-K truncation
    if len(ys) > max_kp:
        ys, xs, scores = ys[:max_kp], xs[:max_kp], scores[:max_kp]

    # Step 7: bilinear interpolation for descriptors
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

    # Step 8: L2 normalize
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    descriptors = descriptors / norms

    return descriptors.astype(np.float32)


# ──────────────────────────────────────────────
# 流程一: argparse 参数接口
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Extract SuperPoint descriptors for BoW vocabulary training')

    p.add_argument('image_dir', type=str,
                   help='Directory containing PNG/JPG images')
    p.add_argument('-o', '--output', type=str, default='sp_descriptors.bin',
                   help='Output binary file path (default: sp_descriptors.bin)')
    p.add_argument('--model', type=str,
                   default='/home/yuan/ORB_SLAM3/Models/superpoint_v1.onnx',
                   help='Path to SuperPoint ONNX model')
    p.add_argument('--max_keypoints', type=int, default=200,
                   help='Max keypoints per image (default: 200)')
    p.add_argument('--skip_frames', type=int, default=5,
                   help='Process every N-th image (default: 5)')
    p.add_argument('--conf_thresh', type=float, default=0.015,
                   help='Minimum keypoint confidence (default: 0.015)')
    p.add_argument('--nms_radius', type=int, default=4,
                   help='NMS suppression radius in pixels (default: 4)')

    return p.parse_args()


# ──────────────────────────────────────────────
# 流程三: 全局跳帧遍历 + 流程四: 二进制无损写入
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # Collect images
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.png')))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))
    if not image_paths:
        print(f"ERROR: No images found in {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    # Skip frames: only sample every N-th image for visual diversity
    sampled = image_paths[::args.skip_frames]

    print(f"Dataset:  {args.image_dir}")
    print(f"Total images:    {len(image_paths)}")
    print(f"Skip frames:     {args.skip_frames}  ->  sampled: {len(sampled)}")
    print(f"Max keypoints:   {args.max_keypoints}")
    print(f"Conf threshold:  {args.conf_thresh}")
    print(f"NMS radius:      {args.nms_radius}")
    print(f"Model:           {args.model}")
    print()

    # Load ONNX model
    print("Loading ONNX model...", flush=True)
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    print("Model loaded.", flush=True)

    # Extract descriptors
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
                                   conf_thresh=args.conf_thresh,
                                   nms_radius=args.nms_radius,
                                   max_kp=args.max_keypoints)

        n_kp = len(desc)
        all_num_kp.append(n_kp)
        if n_kp > 0:
            all_descriptors.append(desc)
            total_kp += n_kp

        print(f"  [{i+1:4d}/{len(sampled)}] {os.path.basename(img_path)}: "
              f"{n_kp:4d} kp  (global idx {i * args.skip_frames})",
              flush=True)

    # Write binary file
    # Format: num_images(i32) + counts(i32×N) + descriptors(float32×total×256)
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
