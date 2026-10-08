import unittest
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator

from detection import UAVDetectionHead, build_uav_detector


class TinyPyramid(nn.Module):
    """Small backbone to exercise the real head, assignment, losses and NMS."""
    out_channels = 32

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 32, 1)

    def forward(self, x):
        x = self.projection(x)
        return OrderedDict((str(s), torch.nn.functional.avg_pool2d(x, s))
                           for s in (4, 8, 16, 32))


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def make_detector(self):
        return FCOS(
            TinyPyramid(), num_classes=3,
            head=UAVDetectionHead(32, 3, num_convs=1),
            anchor_generator=AnchorGenerator(((4,), (8,), (16,), (32,)), ((1.0,),) * 4),
            min_size=64, max_size=96, score_thresh=0.0,
            detections_per_img=20, topk_candidates=30,
        )

    def test_head_shapes_and_validation(self):
        head = UAVDetectionHead(32, 3, num_convs=1)
        features = [torch.randn(2, 32, h, w) for h, w in
                    ((16, 24), (8, 12), (4, 6), (2, 3))]
        outputs = head(features)
        count = sum(x.shape[-2] * x.shape[-1] for x in features)
        self.assertEqual(outputs['cls_logits'].shape, (2, count, 3))
        self.assertEqual(outputs['bbox_regression'].shape, (2, count, 4))
        self.assertEqual(outputs['bbox_ctrness'].shape, (2, count, 1))
        self.assertTrue((outputs['bbox_regression'] >= 0).all())
        with self.assertRaises(ValueError):
            head(features[:-1])
        with self.assertRaises(ValueError):
            UAVDetectionHead(48, 3)

    def test_training_with_objects_and_empty_image(self):
        net = self.make_detector().train()
        images = [torch.rand(3, 64, 96), torch.rand(3, 64, 80)]
        targets = [
            {'boxes': torch.tensor([[12., 12., 28., 28.], [32., 8., 80., 56.]]),
             'labels': torch.tensor([0, 2])},
            {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.int64)},
        ]
        losses = net(images, targets)
        self.assertEqual(set(losses), {'classification', 'bbox_regression', 'bbox_ctrness'})
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        self.assertGreater(losses['bbox_regression'].item(), 0)
        sum(losses.values()).backward()
        gradient = net.head.context[0].contrast.local_1[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_all_empty_targets(self):
        net = self.make_detector().train()
        losses = net([torch.rand(3, 64, 96)],
                     [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.int64)}])
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        sum(losses.values()).backward()

    def test_inference_original_coordinates(self):
        net = self.make_detector().eval()
        with torch.no_grad():
            prediction = net([torch.rand(3, 75, 113)])[0]
        boxes = prediction['boxes']
        self.assertGreater(len(boxes), 0)
        self.assertLessEqual(len(boxes), 20)
        self.assertTrue(torch.isfinite(boxes).all())
        self.assertTrue((boxes >= 0).all())
        self.assertTrue((boxes[:, [0, 2]] <= 113).all())
        self.assertTrue((boxes[:, [1, 3]] <= 75).all())
        self.assertTrue(((prediction['labels'] >= 0) & (prediction['labels'] < 3)).all())

    def test_real_resnext_detector(self):
        net = build_uav_detector(3, channels=32, min_size=64, max_size=96).eval()
        with torch.no_grad():
            result = net([torch.rand(3, 64, 96)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['boxes'].shape[-1], 4)


if __name__ == '__main__':
    unittest.main()
