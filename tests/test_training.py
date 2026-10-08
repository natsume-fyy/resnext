import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

import test_detection
from training import SizeEvaluator, evaluate_by_size, fit, format_epoch


def records(boxes, prediction=False):
    result = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
              'labels': torch.zeros(len(boxes), dtype=torch.int64)}
    if prediction:
        result['scores'] = torch.full((len(boxes),), 0.9)
    return result


class SizeMetricsTests(unittest.TestCase):
    def compute(self, boxes, predicted):
        evaluator = SizeEvaluator(1)
        evaluator.update([torch.zeros(3, 256, 256)], [records(boxes)],
                         [records(predicted, True)])
        return evaluator.compute()

    def test_perfect_and_missed_sizes(self):
        boxes = [[0, 0, 16, 16], [40, 40, 100, 100], [120, 120, 240, 240]]
        perfect = self.compute(boxes, boxes)
        for key in ('AP_small', 'AP_medium', 'AP_large', 'AR_small', 'AR_medium', 'AR_large'):
            self.assertAlmostEqual(perfect[key], 1.0)
        missed = self.compute(boxes, boxes[1:])
        self.assertEqual(missed['AP_small'], 0.0)
        self.assertEqual(missed['AR_small'], 0.0)
        self.assertAlmostEqual(missed['AP_medium'], 1.0)
        self.assertAlmostEqual(missed['AP_large'], 1.0)

    def test_empty_predictions_and_absent_size(self):
        metrics = self.compute([[0, 0, 16, 16]], [])
        self.assertEqual(metrics['AP_small'], 0)
        self.assertIsNone(metrics['AP_medium'])
        self.assertIsNone(metrics['AR_large'])
        self.assertIn('N/A', format_epoch(1, {'total': 1.0}, metrics))
        self.assertTrue(all(v is None for v in self.compute([], []).values()))

    def test_coco_boundary_convention(self):
        metrics = self.compute([[0, 0, 32, 32]], [[0, 0, 32, 32]])
        # COCO includes exact thresholds in both adjacent ranges.
        self.assertAlmostEqual(metrics['AP_small'], 1.0)
        self.assertAlmostEqual(metrics['AP_medium'], 1.0)
        self.assertIsNone(metrics['AP_large'])

    def test_false_positive_on_empty_image_is_counted(self):
        evaluator = SizeEvaluator(1)
        wrong = records([[0, 0, 16, 16]], True)
        wrong['scores'][0] = 0.99
        evaluator.update([torch.zeros(3, 64, 64)] * 2,
                         [records([[0, 0, 16, 16]]), records([])],
                         [records([[0, 0, 16, 16]], True), wrong])
        self.assertLess(evaluator.compute()['AP_small'], 0.6)

    def test_epoch_output_csv_checkpoint_and_restore_mode(self):
        torch.set_num_threads(2)
        model = test_detection.DetectionTests().make_detector()
        loader = [([torch.rand(3, 64, 96)], [records([[12, 12, 28, 28]])])]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        original_threshold = model.score_thresh
        with tempfile.TemporaryDirectory() as directory:
            captured = io.StringIO()
            with redirect_stdout(captured):
                history = fit(model, loader, loader, optimizer, 3, 2, output_dir=directory)
            self.assertEqual(len(history), 2)
            for label in ('大目标', '中目标', '小目标', 'Epoch 2'):
                self.assertIn(label, captured.getvalue())
            with (Path(directory) / 'metrics.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertIn('AP_small', rows[0])
            self.assertEqual(rows[0]['AP_large'], '')
            checkpoint = torch.load(Path(directory) / 'last.pth', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 2)
        self.assertTrue(model.training)
        self.assertEqual(model.score_thresh, original_threshold)
        with self.assertRaises(ValueError):
            evaluate_by_size(model, [], 3)
        self.assertTrue(model.training)
        self.assertEqual(model.score_thresh, original_threshold)


if __name__ == '__main__':
    unittest.main()
