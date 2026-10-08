"""Run `python train.py` on the AutoDL machine containing HazyDet_RFDETR."""

import torch
from torch.utils.data import DataLoader

from detection import build_uav_detector
from detection_dataset import CocoDetectionDataset
from training import detection_collate, fit


DATASET_DIR = '/root/autodl-tmp/HazyDet_RFDETR'
OUTPUT_DIR = '/root/autodl-tmp/rf-detr-output/hazydet_small_test'


def train(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR, epochs=1,
          batch_size=2, grad_accum_steps=2, lr=1e-4, device='cuda',
          num_workers=4, use_ema=True, checkpoint_interval=1,
          backbone_path=None, min_size=800, max_size=1333):
    from pathlib import Path

    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install CUDA-enabled PyTorch or set device="cpu".')
    train_set = CocoDetectionDataset(Path(dataset_dir) / 'train', training=True)
    val_set = CocoDetectionDataset(Path(dataset_dir) / 'valid', categories=train_set.categories)
    loader_options = dict(batch_size=batch_size, num_workers=num_workers,
                          collate_fn=detection_collate, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    model = build_uav_detector(len(train_set.categories), backbone_path=backbone_path,
                               min_size=min_size, max_size=max_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print(f'Train images: {len(train_set)}, validation images: {len(val_set)}', flush=True)
    print(f'Classes (original COCO ID -> name): {train_set.categories}', flush=True)
    print(f'Device: {device}; effective batch size: {batch_size * grad_accum_steps}; '
          f'EMA: {use_ema}; output: {output_dir}', flush=True)
    return fit(model, train_loader, val_loader, optimizer, len(train_set.categories), epochs,
               device=device, output_dir=output_dir, grad_accum_steps=grad_accum_steps,
               use_ema=use_ema, checkpoint_interval=checkpoint_interval,
               metadata={'categories': train_set.categories,
                         'category_to_label': train_set.category_to_label,
                         'min_size': min_size, 'max_size': max_size})


def main():
    train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=1,
        batch_size=2,
        grad_accum_steps=2,
        lr=1e-4,
        device='cuda',
        num_workers=4,
        use_ema=True,
        checkpoint_interval=1,
    )


if __name__ == '__main__':
    main()
