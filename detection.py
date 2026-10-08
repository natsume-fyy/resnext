"""Context-aware FCOS detection for UAV imagery.

The head accepts P2-P5 features, ordered from high to low resolution.
Weather robustness is a training/evaluation objective, not a built-in guarantee.
"""

from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.fcos import FCOSHead
from torchvision.ops import FeaturePyramidNetwork

from model import DenseContrastModule


class ContextEnhancement(nn.Module):
    """Retain local appearance alongside the original six context differences."""

    def __init__(self, channels):
        super().__init__()
        self.contrast = DenseContrastModule(channels)
        # UAV training often uses small batches: avoid batch-dependent statistics.
        for branch in (self.contrast.local_1, self.contrast.context_1,
                       self.contrast.context_2, self.contrast.context_3):
            branch[1] = nn.GroupNorm(1, channels // 8)
        self.project = nn.Conv2d(6 * (channels // 8), channels, 1)
        self.norm = nn.GroupNorm(32, channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.norm(self.project(self.contrast(x))))


class UAVDetectionHead(FCOSHead):
    """Standalone FCOS-compatible head with context enhancement per level.

    Inputs: list of NCHW tensors, each with ``in_channels`` channels.
    Outputs: cls_logits [N, locations, K], bbox_regression [N, locations, 4]
    (nonnegative l/t/r/b distances normalized by feature stride), and
    bbox_ctrness [N, locations, 1]. Raw outputs need FCOS decoding and NMS.
    Classes use contiguous IDs 0..num_classes-1 (no background class).
    """

    def __init__(self, in_channels=256, num_classes=1, num_levels=4, num_convs=4):
        if in_channels < 32 or in_channels % 32:
            raise ValueError('in_channels must be a positive multiple of 32')
        if num_classes < 1 or num_levels < 1 or num_convs < 1:
            raise ValueError('num_classes, num_levels and num_convs must be positive')
        super().__init__(in_channels, num_anchors=1, num_classes=num_classes,
                         num_convs=num_convs)
        self.in_channels = in_channels
        self.context = nn.ModuleList(
            ContextEnhancement(in_channels) for _ in range(num_levels)
        )

    def forward(self, features):
        if len(features) != len(self.context):
            raise ValueError(f'Expected {len(self.context)} feature levels, got {len(features)}')
        for feature in features:
            if feature.ndim != 4 or feature.shape[1] != self.in_channels:
                raise ValueError(f'Each feature must have shape [N, {self.in_channels}, H, W]')
        return super().forward([block(x) for block, x in zip(self.context, features)])


class ResNeXtPyramid(nn.Module):
    """Original ResNeXt-101 features plus P2-P5 FPN (strides 4, 8, 16, 32)."""

    def __init__(self, backbone_path=None, out_channels=256):
        super().__init__()
        from backbone.resnext.resnext101_regular import ResNeXt101

        # The original module exposes a global network; isolate each detector.
        self.body = deepcopy(ResNeXt101(None))
        if backbone_path is not None:
            from backbone.resnext.resnext_101_32x4d_ import resnext_101_32x4d

            original = deepcopy(resnext_101_32x4d)
            original.load_state_dict(torch.load(backbone_path, map_location='cpu',
                                               weights_only=True))
            layers = list(original.children())
            self.body.layer0 = nn.Sequential(*layers[:3])
            self.body.layer1 = nn.Sequential(*layers[3:5])
            self.body.layer2, self.body.layer3, self.body.layer4 = layers[5:8]
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body.layer0(x)
        features = OrderedDict()
        for name in ('layer1', 'layer2', 'layer3', 'layer4'):
            x = getattr(self.body, name)(x)
            features[name] = x
        return self.fpn(features)


def build_uav_detector(num_classes, backbone_path=None, channels=256,
                       min_size=800, max_size=1333, **kwargs):
    """Build a trainable detector with FCOS assignment, losses, decoding and NMS.

    images: list of float RGB tensors [3,H,W] in [0,1], before normalization.
    Training targets: boxes float [M,4] in absolute xyxy image coordinates;
    labels int64 [M], IDs 0..num_classes-1. Empty targets are supported.
    train(): returns classification, bbox_regression and bbox_ctrness losses.
    eval(): returns boxes, labels and scores for each original-sized image.
    No pretrained weights are downloaded automatically.
    """
    head = UAVDetectionHead(channels, num_classes)
    backbone = ResNeXtPyramid(backbone_path, channels)
    # FCOS requires one square reference anchor per location, sized by stride.
    anchors = AnchorGenerator(sizes=((4,), (8,), (16,), (32,)),
                              aspect_ratios=((1.0,),) * 4)
    return FCOS(backbone, num_classes, anchor_generator=anchors, head=head,
                min_size=min_size, max_size=max_size, **kwargs)
