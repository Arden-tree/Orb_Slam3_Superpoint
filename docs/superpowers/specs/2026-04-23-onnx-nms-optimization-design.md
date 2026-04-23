# SuperPoint ONNX Model NMS Optimization Design

## Goal

Optimize SuperPoint frontend from ~2.3s/frame to ~12ms/frame by:
1. Moving softmax + reshape into ONNX model (NPU acceleration)
2. Replacing O(N^2) brute-force NMS with cv::dilate (~1ms)
3. Moving L2 normalization into ONNX model (NPU)

## Architecture

```
NPU (ONNX model):
  Image(1,1,H,W) -> VGG Encoder -> Score Head -> Softmax(64ch)
    -> Slice(drop dustbin) -> Reshape(8x8 -> H,W)
    -> Descriptor Head -> Normalize(L2)
  Output: heatmap(H,W) + dense_desc(256, H/8, W/8)

CPU post-processing (~1-2ms):
  cv::dilate(kernel=9x9) -> (heatmap == local_max) -> threshold filter
    -> extract coords -> bilinear interpolation -> Top-K
  Output: keypoints + descriptors(N, 256)
```

## Icraft NPU Operator Support (v3.33.1)

Supported: Conv2d, MaxPool2d, Softmax, Reshape, Transpose, Slice, Normalize, ReLU
NOT supported: Where, NonZero, TopK, GridSample, Sub

Strategy: put all supported ops in ONNX, leave unsupported ops on CPU.

## Changes

### 1. export_superpoint_onnx.py
- Add softmax(dim=1)[:, :-1] to forward()
- Add reshape: (B, 64, H/8, W/8) -> (B, H, W)
- Add L2 normalize on descriptor head
- Output names: 'heatmap', 'desc' (was: 'semi', 'desc')
- Keep dynamic axes for height/width

### 2. SuperPointExtractor.cc
- Remove: softmax loop (L162-197), brute-force NMS (L199-255)
- Add: cv::dilate-based NMS (3 lines)
- Simplify: only threshold -> extract coords -> bilinear interpolation -> Top-K
- Add: SetIntraOpNumThreads(4) for CPU fallback
- Remove: per-frame stderr logging

### 3. Backward compatibility
- ORB mode completely unaffected
- SuperPoint mode: ONNX model path unchanged (superpoint_v1.onnx)
- YAML config: same parameters, new defaults for TH_LOW/TH_HIGH

## Performance Target

| Stage | Current | Target |
|-------|---------|--------|
| ONNX inference (CPU) | ~300ms | ~10ms (NPU) |
| Softmax + Reshape (CPU) | ~200ms | 0ms (in NPU) |
| NMS | ~1600ms | ~1ms (cv::dilate) |
| Interpolation + L2 norm | ~100ms | ~1ms (L2 in NPU) |
| **Total** | **~2.3s** | **~12ms** |
