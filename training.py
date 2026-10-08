"""Single-process detector training with per-epoch COCO size metrics."""

import csv
import io
import math
from contextlib import redirect_stdout
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')


def detection_collate(batch):
    """Keep variable-sized images and target dictionaries as lists."""
    images, targets = zip(*batch)
    return list(images), list(targets)


class SizeEvaluator:
    """COCO bbox metrics using input-image box area (before model resizing).

    Class IDs are 0..num_classes-1. A missing GT size group yields None.
    Optional iscrowd flags are respected; segmentation areas are not used.
    """

    def __init__(self, num_classes):
        if num_classes < 1:
            raise ValueError('num_classes must be positive')
        self.num_classes = num_classes
        self.images = []
        self.annotations = []
        self.predictions = []

    def _records(self, data, image_id, prediction=False):
        boxes = data['boxes'].detach().cpu()
        labels = data['labels'].detach().cpu()
        if boxes.ndim != 2 or boxes.shape[1] != 4 or labels.shape != (len(boxes),):
            raise ValueError('Expected boxes [N,4] and labels [N]')
        if labels.dtype != torch.int64:
            raise ValueError('labels must be int64')
        sizes = boxes[:, 2:] - boxes[:, :2]
        invalid_size = (sizes < 0).any() if prediction else (sizes <= 0).any()
        if not torch.isfinite(boxes).all() or invalid_size:
            raise ValueError('Boxes must be finite with valid xyxy coordinates')
        if ((labels < 0) | (labels >= self.num_classes)).any():
            raise ValueError('Class IDs must be in 0..num_classes-1')
        scores = data.get('scores') if prediction else None
        if prediction:
            if scores is None or scores.shape != labels.shape or not torch.isfinite(scores).all():
                raise ValueError('Expected finite scores [N]')
            scores = scores.detach().cpu()
        crowd = data.get('iscrowd', torch.zeros(len(boxes), dtype=torch.int64)).detach().cpu()
        if crowd.shape != labels.shape:
            raise ValueError('Expected iscrowd [N]')
        records = []
        for i, (box, label) in enumerate(zip(boxes.tolist(), labels.tolist())):
            x1, y1, x2, y2 = box
            record = {'image_id': image_id, 'category_id': label + 1,
                      'bbox': [x1, y1, x2 - x1, y2 - y1],
                      'area': (x2 - x1) * (y2 - y1)}
            if prediction:
                record['score'] = float(scores[i])
            else:
                record['iscrowd'] = int(crowd[i])
            records.append(record)
        return records

    def update(self, images, targets, predictions):
        if not len(images) == len(targets) == len(predictions):
            raise ValueError('Image, target and prediction counts must match')
        for image, target, prediction in zip(images, targets, predictions):
            image_id = len(self.images) + 1
            gt = self._records(target, image_id)
            dt = self._records(prediction, image_id, prediction=True)
            self.images.append({'id': image_id, 'height': image.shape[-2],
                                'width': image.shape[-1]})
            for record in gt:
                record['id'] = len(self.annotations) + 1
                self.annotations.append(record)
            self.predictions.extend(dt)

    def compute(self):
        if not self.images:
            raise ValueError('Validation loader is empty')
        categories = [{'id': i + 1, 'name': str(i)} for i in range(self.num_classes)]
        # Construct the detection COCO object directly: loadRes rejects [] in
        # some versions, while zero detections are valid during early training.
        with redirect_stdout(io.StringIO()):
            gt, dt = COCO(), COCO()
            gt.dataset = {'images': self.images, 'categories': categories,
                          'annotations': self.annotations}
            dt.dataset = {'images': self.images, 'categories': categories,
                          'annotations': [dict(p, id=i + 1, iscrowd=0)
                                          for i, p in enumerate(self.predictions)]}
            gt.createIndex()
            dt.createIndex()
            evaluator = COCOeval(gt, dt, 'bbox')
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        return {name: (float(value) if value >= 0 else None)
                for name, value in zip(METRIC_NAMES, evaluator.stats)}


@torch.inference_mode()
def evaluate_by_size(model, loader, num_classes, device='cpu'):
    evaluator = SizeEvaluator(num_classes)
    was_training = model.training
    old_threshold = getattr(model, 'score_thresh', None)
    try:
        model.eval()
        if old_threshold is not None:
            model.score_thresh = 0.001
        for images, targets in loader:
            predictions = model([image.to(device) for image in images])
            evaluator.update(images, targets, predictions)
        return evaluator.compute()
    finally:
        model.train(was_training)
        if old_threshold is not None:
            model.score_thresh = old_threshold


def train_one_epoch(model, loader, optimizer, device='cpu'):
    model.train()
    totals, num_images = {}, 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        total = sum(losses.values())
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite training loss')
        total.backward()
        optimizer.step()
        for key, value in dict(losses, total=total).items():
            totals[key] = totals.get(key, 0.0) + value.detach().item() * len(images)
        num_images += len(images)
    if not num_images:
        raise ValueError('Training loader is empty')
    return {key: value / num_images for key, value in totals.items()}


def format_epoch(epoch, losses, metrics):
    def percent(value):
        return 'N/A' if value is None or not math.isfinite(value) else f'{value * 100:.2f}'

    lines = [f'Epoch {epoch} | loss={losses["total"]:.4f} | '
             f'AP50={percent(metrics["AP50"])} | AP75={percent(metrics["AP75"])}',
             '目标尺度       AP@[.50:.95] (%)    AR@100 (%)']
    for label, suffix in [('整体', ''), ('大目标', '_large'),
                          ('中目标', '_medium'), ('小目标', '_small')]:
        ap_key = 'AP' + suffix
        ar_key = 'AR' + suffix if suffix else 'AR100'
        lines.append(f'{label:<8} {percent(metrics[ap_key]):>12} {percent(metrics[ar_key]):>17}')
    return '\n'.join(lines)


def fit(model, train_loader, val_loader, optimizer, num_classes, epochs,
        device='cpu', output_dir='runs/train', start_epoch=1):
    """Train then validate every epoch; save metrics.csv and last.pth.

    Move the model to device BEFORE creating the optimizer. Loaders must be
    re-iterable, with detection_collate; validation must cover the full split.
    Existing CSV files are never overwritten. For resume, use a new output_dir
    and explicitly restore model/optimizer and set start_epoch.
    """
    if epochs < 1 or start_epoch < 1:
        raise ValueError('epochs and start_epoch must be positive')
    if num_classes < 1:
        raise ValueError('num_classes must be positive')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    with (output_dir / 'metrics.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = None
        for epoch in range(start_epoch, start_epoch + epochs):
            losses = train_one_epoch(model, train_loader, optimizer, device)
            metrics = evaluate_by_size(model, val_loader, num_classes, device)
            print(format_epoch(epoch, losses, metrics), flush=True)
            row = {'epoch': epoch, **{'loss_' + k: v for k, v in losses.items()}, **metrics}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'metrics': metrics,
                        'num_classes': num_classes}, output_dir / 'last.pth')
            history.append(row)
    return history
