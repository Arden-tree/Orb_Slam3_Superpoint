#!/usr/bin/env python3
"""
Visualize SuperPoint keypoints on TUM dataset images in real-time.

Usage:
  conda activate superpoint
  python3 Tools/visualize_sp_keypoints.py <image_dir>
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import onnxruntime as ort


def softmax_64ch(semi):
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
    kernel_size = 2 * nms_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    local_max = cv2.dilate(scores, kernel)
    is_max = (scores == local_max) & (scores > 0)
    ys, xs = np.where(is_max)
    return ys, xs, scores[ys, xs]


def extract_keypoints(session, img_float, conf_thresh=0.015, nms_radius=4, max_kp=1000):
    H, W = img_float.shape
    input_tensor = img_float[np.newaxis, np.newaxis, :, :].astype(np.float32)
    semi, desc = session.run(None, {'input': input_tensor})

    semi = semi[0]
    desc = desc[0]
    featH, featW = desc.shape[1], desc.shape[2]

    heatmap = softmax_64ch(semi)
    ys, xs, scores = nms_dilate(heatmap, nms_radius)
    if len(ys) == 0:
        return np.array([]), np.array([]), np.array([])

    mask = scores >= conf_thresh
    ys, xs, scores = ys[mask], xs[mask], scores[mask]

    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    if len(ys) > max_kp:
        ys, xs, scores = ys[:max_kp], xs[:max_kp], scores[:max_kp]

    return ys, xs, scores


def draw_keypoints(img_color, ys, xs, scores):
    vis = img_color.copy()
    if len(ys) == 0:
        return vis

    # Normalize scores for color mapping
    smin, smax = scores.min(), scores.max()
    if smax > smin:
        norm_scores = (scores - smin) / (smax - smin)
    else:
        norm_scores = np.ones_like(scores)

    for i in range(len(ys)):
        x, y = int(xs[i]), int(ys[i])
        # Green=high confidence, Red=low confidence
        g = int(255 * norm_scores[i])
        r = int(255 * (1 - norm_scores[i]))
        color = (r, g, 0)
        cv2.circle(vis, (x, y), 3, color, -1)

    return vis


def main():
    parser = argparse.ArgumentParser(description='Visualize SuperPoint keypoints')
    parser.add_argument('image_dir', type=str)
    parser.add_argument('--model', type=str,
                        default='/home/yuan/ORB_SLAM3/Models/superpoint_v1.onnx')
    parser.add_argument('--conf_thresh', type=float, default=0.015)
    parser.add_argument('--nms_radius', type=int, default=4)
    parser.add_argument('--max_kp', type=int, default=1000)
    parser.add_argument('--skip', type=int, default=1, help='Show every N-th image')
    parser.add_argument('--pause_ms', type=int, default=50, help='Pause between frames (ms, 0=wait for key)')
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.png')))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))
    if not image_paths:
        print(f"No images found in {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    print(f"Images: {len(image_paths)}, skip={args.skip}, max_kp={args.max_kp}")
    print("Controls: SPACE=pause, Q/ESC=quit, +/-=speed")
    print()

    idx = 0
    paused = False
    while idx < len(image_paths):
        img_path = image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            idx += args.skip
            continue

        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        H, W = img.shape
        img_float = (img.astype(np.float32) / 255.0)[:((H // 8) * 8), :((W // 8) * 8)]

        ys, xs, scores = extract_keypoints(session, img_float,
                                           conf_thresh=args.conf_thresh,
                                           nms_radius=args.nms_radius,
                                           max_kp=args.max_kp)

        vis = draw_keypoints(img_color, ys, xs, scores)

        # Info overlay
        info = f"Frame {idx}/{len(image_paths)}  KP: {len(ys)}  Score: {scores.min():.3f}-{scores.max():.3f}"
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, os.path.basename(img_path), (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow('SuperPoint Keypoints', vis)

        key = cv2.waitKey(0 if paused else args.pause_ms) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('+') or key == ord('='):
            idx = max(0, idx - args.skip * 5)
        elif key == ord('-'):
            idx = min(len(image_paths) - 1, idx + args.skip * 5)
        else:
            idx += args.skip

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
