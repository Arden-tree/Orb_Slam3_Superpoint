#!/usr/bin/env python3
"""
Extract SuperPoint descriptors from TUM dataset images using ONNX Runtime.
Output: binary file with all descriptors for vocabulary training.

Output format:
  - Header: num_images (int32), image0_num_kp (int32), image0_num_kp (int32), ...
  - For each image: num_kp descriptors, each 256 floats (256 * 4 = 1024 bytes)
  - All float32, native byte order.

Usage:
  conda activate superpoint
  python3 Tools/extract_sp_descriptors.py <image_dir> <output.bin> [max_images]
"""

import os
import sys
import glob
import struct
import numpy as np
import onnxruntime as ort
import cv2


def nms_fast(scores, NMS_RADIUS=4, max_kp=2000):
    """
    Fast NMS using grid-based suppression with OpenCV maxPool.
    Returns indices of keypoints sorted by score descending, up to max_kp.
    """
    H, W = scores.shape

    # Pad to avoid border effects
    pad = NMS_RADIUS
    padded = np.pad(scores, pad, mode='constant', constant_values=0)

    # Use cv2.dilate as a local max filter (faster than manual grid)
    kernel_size = 2 * NMS_RADIUS + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    local_max = cv2.dilate(scores, kernel)

    # Keep only local maxima (ties broken by position)
    is_max = (scores == local_max) & (scores > 0)
    ys, xs = np.where(is_max)
    scores_sel = scores[ys, xs]

    # Sort by score descending
    order = np.argsort(-scores_sel)
    ys = ys[order][:max_kp]
    xs = xs[order][:max_kp]

    return ys, xs, scores_sel[order][:max_kp]


def softmax_64ch(semi):
    """
    Dustbin-free softmax + reshape to full-resolution heatmap.

    Args:
        semi: (65, H/8, W/8) raw logits
    Returns:
        heatmap: (H, W) full-resolution score map
    """
    cells = semi[:64]  # (64, featH, featW)

    # Softmax over 64 channels
    cells = cells - cells.max(axis=0, keepdims=True)
    exp_cells = np.exp(cells)
    prob = exp_cells / exp_cells.sum(axis=0, keepdims=True)  # (64, featH, featW)

    # Reshape 64 channels -> 8x8 grid -> full resolution
    featH, featW = prob.shape[1], prob.shape[2]
    H = featH * 8
    W = featW * 8
    heatmap = np.zeros((H, W), dtype=np.float32)

    for dy in range(8):
        for dx in range(8):
            c = dy * 8 + dx
            heatmap[dy::8, dx::8] = prob[c]

    return heatmap


def extract_descriptors(session, img_float, conf_threshold=0.015, nms_radius=4, max_kp=2000):
    """
    Extract SuperPoint descriptors from a pre-loaded float image.

    Args:
        session: ONNX Runtime session
        img_float: (H, W) float32, grayscale [0,1], already padded to multiple of 8

    Returns:
        descriptors: (N, 256) float32, L2-normalized
    """
    H, W = img_float.shape

    # ONNX inference
    input_tensor = img_float[np.newaxis, np.newaxis, :, :].astype(np.float32)
    semi, desc = session.run(None, {'input': input_tensor})

    semi = semi[0]  # (65, H/8, W/8)
    desc = desc[0]  # (256, H/8, W/8)

    # Softmax + reshape
    heatmap = softmax_64ch(semi)

    # NMS with thresholding
    ys, xs, scores = nms_fast(heatmap, NMS_RADIUS=nms_radius, max_kp=max_kp)

    if len(ys) == 0:
        return np.zeros((0, 256), dtype=np.float32)

    # Apply confidence threshold (after NMS for speed)
    mask = scores >= conf_threshold
    if not np.any(mask):
        return np.zeros((0, 256), dtype=np.float32)
    ys = ys[mask]
    xs = xs[mask]

    # Bilinear interpolation for descriptors
    featH, featW = desc.shape[1], desc.shape[2]

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
                   desc[:, y1, x1] * wd[np.newaxis, :])

    descriptors = descriptors.T  # (N, 256)

    # L2 normalize
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    descriptors = descriptors / norms

    return descriptors.astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <image_dir> <output.bin> [max_images]")
        sys.exit(1)

    image_dir = sys.argv[1]
    output_path = sys.argv[2]
    max_images = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # Load ONNX model
    model_path = '/home/yuan/ORB_SLAM3/Models/superpoint_v1.onnx'
    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Collect images
    images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    if not images:
        images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    if not images:
        print(f"ERROR: No images found in {image_dir}")
        sys.exit(1)

    if max_images > 0:
        images = images[:max_images]

    print(f"Found {len(images)} images in {image_dir}")

    # Extract descriptors
    all_num_kp = []
    all_descriptors = []
    total_kp = 0

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            all_num_kp.append(0)
            continue

        H, W = img.shape
        img_float = (img.astype(np.float32) / 255.0)[:((H // 8) * 8), :((W // 8) * 8)]

        desc = extract_descriptors(session, img_float,
                                   conf_threshold=0.015,
                                   nms_radius=4,
                                   max_kp=2000)

        n_kp = len(desc)
        all_num_kp.append(n_kp)
        if n_kp > 0:
            all_descriptors.append(desc)
            total_kp += n_kp

        if (i + 1) % 50 == 0 or i == 0 or i == len(images) - 1:
            print(f"  [{i+1:3d}/{len(images)}] {os.path.basename(img_path)}: {n_kp} kp", flush=True)

    # Save to binary file
    print(f"\nTotal keypoints: {total_kp} from {len(images)} images")

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<i', len(images)))
        for n in all_num_kp:
            f.write(struct.pack('<i', n))
        for desc in all_descriptors:
            f.write(desc.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"Saved to {output_path} ({file_size / 1024 / 1024:.1f} MB)")
    print("Done!")


if __name__ == '__main__':
    main()
