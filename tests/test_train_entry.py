import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import test_detection
from detection_dataset import CocoDetectionDataset
from train import train
from training import ModelEMA, train_one_epoch


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, images, targets):
        return {'loss': torch.stack([(self.weight - x).square() for x in images]).mean()}


class TrainEntryTests(unittest.TestCase):
    def test_accumulation_matches_large_batches_including_tail(self):
        model, reference = ScalarModel(), ScalarModel()
        ema = ModelEMA(model)
        micro = [([torch.tensor(x) for x in group], [{}] * len(group))
                 for group in ([1., 2.], [3.], [4.])]
        large = [([torch.tensor(x) for x in group], [{}] * len(group))
                 for group in ([1., 2., 3.], [4.])]
        train_one_epoch(model, micro, torch.optim.SGD(model.parameters(), lr=0.1),
                        grad_accum_steps=2, ema=ema)
        train_one_epoch(reference, large, torch.optim.SGD(reference.parameters(), lr=0.1))
        torch.testing.assert_close(model.weight, reference.weight)
        self.assertEqual(ema.updates, 2)
        self.assertFalse(ema.model.weight.requires_grad)

    def test_coco_entry_with_ema_and_checkpoint(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, count in [('train', 3), ('valid', 1)]:
                folder = root / split
                folder.mkdir()
                images, annotations = [], []
                for i in range(count):
                    name = f'{i}.png'
                    Image.new('RGB', (96, 64)).save(folder / name)
                    images.append({'id': i, 'file_name': name, 'width': 96, 'height': 64})
                    annotations.append({'id': i, 'image_id': i, 'category_id': 7,
                                        'bbox': [12, 12, 16, 16], 'iscrowd': 0})
                (folder / '_annotations.coco.json').write_text(json.dumps({
                    'images': images, 'annotations': annotations,
                    'categories': [{'id': 7, 'name': 'car'}, {'id': 19, 'name': 'truck'},
                                   {'id': 30, 'name': 'bus'}]}), encoding='utf-8')
            dataset = CocoDetectionDataset(root / 'train')
            image, target = dataset[0]
            self.assertEqual(image.shape, (3, 64, 96))
            self.assertEqual(target['labels'].tolist(), [0])
            self.assertEqual(target['boxes'].tolist(), [[12., 12., 28., 28.]])
            with patch('train.build_uav_detector', return_value=test_detection.DetectionTests().make_detector()):
                with redirect_stdout(io.StringIO()):
                    history = train(root, root / 'output', device='cpu', num_workers=0)
            self.assertEqual(history[0]['evaluated_weights'], 'ema')
            checkpoint = torch.load(root / 'output' / 'checkpoint_0001.pth', weights_only=True)
            self.assertIn('ema', checkpoint)
            self.assertEqual(checkpoint['ema_updates'], 1)
            self.assertEqual(checkpoint['metadata']['category_to_label'], {7: 0, 19: 1, 30: 2})
            self.assertTrue((root / 'output' / 'last.pth').is_file())
            self.assertTrue((root / 'output' / 'metrics.csv').is_file())


if __name__ == '__main__':
    unittest.main()
