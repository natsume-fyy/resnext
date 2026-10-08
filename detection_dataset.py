"""Read RF-DETR-style COCO exports for the project's FCOS detector."""

import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class CocoDetectionDataset(Dataset):
    def __init__(self, split_dir, categories=None, training=False):
        self.root = Path(split_dir)
        annotation_path = self.root / '_annotations.coco.json'
        if not annotation_path.is_file():
            raise FileNotFoundError(f'Expected COCO annotations: {annotation_path}')
        data = json.loads(annotation_path.read_text(encoding='utf-8'))
        local_categories = {int(c['id']): c['name'] for c in data['categories']}
        self.categories = dict(sorted((categories if categories is not None else local_categories).items()))
        if not self.categories:
            raise ValueError('No categories found in training annotations')
        for category_id, name in local_categories.items():
            if self.categories.get(category_id) != name:
                raise ValueError(f'Category mismatch in {annotation_path}: {category_id}={name}')
        self.category_to_label = {category_id: index for index, category_id in enumerate(self.categories)}
        self.images = data['images']
        if not self.images:
            raise ValueError(f'No images found in {annotation_path}')
        self.annotations = {image['id']: [] for image in self.images}
        if len(self.annotations) != len(self.images):
            raise ValueError('Duplicate image IDs')
        for annotation in data['annotations']:
            if annotation['category_id'] not in self.category_to_label:
                raise ValueError(f'Unknown category: {annotation["category_id"]}')
            if annotation['image_id'] not in self.annotations:
                raise ValueError(f'Unknown image ID: {annotation["image_id"]}')
            if training and annotation.get('iscrowd', 0):
                continue  # FCOS training does not implement crowd-region ignore.
            self.annotations[annotation['image_id']].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        with Image.open(self.root / info['file_name']) as source:
            image = pil_to_tensor(source.convert('RGB')).float().div_(255)
        height, width = image.shape[-2:]
        if (info.get('height', height), info.get('width', width)) != (height, width):
            raise ValueError(f'Image dimensions differ from annotations: {info["file_name"]}')
        boxes, labels, crowds = [], [], []
        for annotation in self.annotations[info['id']]:
            x, y, w, h = annotation['bbox']
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError(f'Invalid bounding box: {annotation}')
            box = [max(0., x), max(0., y), min(float(width), x + w), min(float(height), y + h)]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f'Bounding box lies outside image: {annotation}')
            boxes.append(box)
            labels.append(self.category_to_label[annotation['category_id']])
            crowds.append(annotation.get('iscrowd', 0))
        return image, {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                       'labels': torch.tensor(labels, dtype=torch.int64),
                       'iscrowd': torch.tensor(crowds, dtype=torch.int64),
                       'image_id': torch.tensor(info['id'], dtype=torch.int64)}
