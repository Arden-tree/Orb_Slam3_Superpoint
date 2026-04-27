#!/usr/bin/env python3
"""
Export SuperPoint backbone to ONNX format (without NMS, keypoint selection).
Outputs raw semi (keypoint scores) and desc (dense descriptors) tensors.
NMS and keypoint selection will be done in C++ (SuperPointExtractor.cc).
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/yuan/SuperGluePretrainedNetwork')
from models.superpoint import SuperPoint


class SuperPointBackbone(nn.Module):
    """SuperPoint backbone that outputs raw score map and dense descriptors.
    NMS and keypoint extraction are done on the C++ side.
    """
    def __init__(self, superpoint_model):
        super().__init__()
        # Copy all layers from the pretrained model
        self.relu = superpoint_model.relu
        self.pool = superpoint_model.pool
        self.conv1a = superpoint_model.conv1a
        self.conv1b = superpoint_model.conv1b
        self.conv2a = superpoint_model.conv2a
        self.conv2b = superpoint_model.conv2b
        self.conv3a = superpoint_model.conv3a
        self.conv3b = superpoint_model.conv3b
        self.conv4a = superpoint_model.conv4a
        self.conv4b = superpoint_model.conv4b
        # Score head (no softmax here - done in C++)
        self.convPa = superpoint_model.convPa
        self.convPb = superpoint_model.convPb
        # Descriptor head
        self.convDa = superpoint_model.convDa
        self.convDb = superpoint_model.convDb

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) float32, grayscale image normalized to [0, 1]
        Returns:
            heatmap: (B, H, W) full-resolution keypoint probability heatmap
            desc: (B, 256, H/8, W/8) L2-normalized dense descriptors
        """
        # Shared encoder
        out = self.relu(self.conv1a(x))
        out = self.relu(self.conv1b(out))
        out = self.pool(out)
        out = self.relu(self.conv2a(out))
        out = self.relu(self.conv2b(out))
        out = self.pool(out)
        out = self.relu(self.conv3a(out))
        out = self.relu(self.conv3b(out))
        out = self.pool(out)
        out = self.relu(self.conv4a(out))
        out = self.relu(self.conv4b(out))

        # Score head: softmax + reshape (superpoint.py L163-166)
        semi = self.convPb(self.relu(self.convPa(out)))
        scores = F.softmax(semi, dim=1)[:, :-1]   # (B, 64, H/8, W/8)
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        heatmap = scores  # (B, H, W)

        # Descriptor head
        desc = self.convDb(self.relu(self.convDa(out)))
        desc = F.normalize(desc, p=2, dim=1)

        return heatmap, desc


def main():
    print("Loading SuperPoint model...")
    model = SuperPoint({})

    print("Creating backbone...")
    backbone = SuperPointBackbone(model)
    backbone.eval()

    # Test with 480x640 (typical TUM dataset resolution)
    H, W = 480, 640
    dummy_input = torch.randn(1, 1, H, W)

    print(f"Input shape: {dummy_input.shape}")

    # Test forward pass
    with torch.no_grad():
        heatmap, desc = backbone(dummy_input)
        print(f"heatmap shape: {heatmap.shape}")  # (1, H, W)
        print(f"desc shape: {desc.shape}")  # (1, 256, H/8, W/8)

    # Export to ONNX
    output_path = '/home/yuan/ORB_SLAM3/Models/superpoint_v1.onnx'
    print(f"Exporting to {output_path}...")

    torch.onnx.export(
        backbone,
        dummy_input,
        output_path,
        opset_version=18,
        input_names=['input'],
        output_names=['heatmap', 'desc'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'heatmap': {0: 'batch', 1: 'height', 2: 'width'},
            'desc': {0: 'batch', 2: 'height', 3: 'width'},
        },
        do_constant_folding=True,
        export_params=True,
    )

    print(f"Exported successfully!")

    # Verify with ONNX Runtime
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    print(f"\nONNX Runtime verification:")
    print(f"  Input name: {sess.get_inputs()[0].name}")
    print(f"  Output names: {[o.name for o in sess.get_outputs()]}")

    # Test inference
    test_img = np.random.randn(1, 1, H, W).astype(np.float32)
    outputs = sess.run(None, {'input': test_img})
    print(f"  heatmap shape: {outputs[0].shape}")
    print(f"  desc shape: {outputs[1].shape}")

    # Compare with PyTorch
    with torch.no_grad():
        heatmap_pt, desc_pt = backbone(torch.from_numpy(test_img))
        heatmap_diff = np.abs(outputs[0] - heatmap_pt.numpy()).max()
        desc_diff = np.abs(outputs[1] - desc_pt.numpy()).max()
        print(f"  Max heatmap diff: {heatmap_diff:.8f}")
        print(f"  Max desc diff: {desc_diff:.8f}")

    if heatmap_diff < 1e-5 and desc_diff < 1e-5:
        print("  PASSED: ONNX output matches PyTorch!")
    else:
        print("  WARNING: ONNX output differs from PyTorch!")

    print("\nDone! Model saved to:", output_path)


if __name__ == '__main__':
    main()
