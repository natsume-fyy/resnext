# UAV 目标检测与原始玻璃表面检测

新增 `detection.py`：把原模型的多尺度上下文差分与注意力用于目标检测头，面向恶劣天气 UAV 目标检测实验。原 `model.py` 中的 `GlassNet` 和 `infer.py` 仍用于玻璃分割。

## 新增目标检测模型

### 在 AutoDL 上直接启动训练

运行 `python train.py`。入口按照给出的 RF-DETR 训练脚本参数配置，实际构建的是本项目 ResNeXt + FCOS UAV 检测模型；没有改用 `RFDETRSmall`。

```python
DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/rf-detr-output/hazydet_small_test"
```

默认参数：`epochs=1`、`batch_size=2`、`grad_accum_steps=2`、`lr=1e-4`、`device="cuda"`、`num_workers=4`、`use_ema=True`、`checkpoint_interval=1`。使用 AdamW（weight_decay=1e-4），通常每 4 张图片更新一次参数；最后不足一个累积窗口也会正确更新。

数据集按 [RF-DETR COCO 导出格式](https://rfdetr.roboflow.com/learn/train/) 读取：

```text
/root/autodl-tmp/HazyDet_RFDETR/
├── train/
│   ├── _annotations.coco.json
│   └── 图片文件...
└── valid/
    ├── _annotations.coco.json
    └── 图片文件...
```

`detection_dataset.py` 将 COCO 的 xywh 框转为 xyxy，按照训练标注中的 categories 自动确定类别数，将原始 ID 映射到从 0 开始的连续编号。验证集沿用同一映射，类别名称不一致时会报错。训练跳过 iscrowd 标注（当前 FCOS 不支持 crowd 忽略区训练），验证保留其忽略语义。图片转为 RGB 并归一到 [0,1]，其余缩放/标准化由模型完成。

每轮在验证集输出整体及大、中、小目标指标。启用 EMA 时，指标来自 EMA 模型，CSV 中 `evaluated_weights=ema`；EMA 衰减系数为 0.999，每次优化器更新后更新。`last.pth` 和每轮的 `checkpoint_0001.pth` 包含原模型 `model`、EMA 权重 `ema`、优化器状态及类别映射。推理要复现 EMA 验证结果，应加载 checkpoint 的 `ema` 字段。

默认没有预训练初始化，仍需自行提供兼容的 `backbone_path` 才能加载骨干权重。输出目录若已有 `metrics.csv` 会报错以保留旧实验；需要新实验时修改 `OUTPUT_DIR`。首次运行前在服务器安装 CUDA 版 PyTorch 和匹配 torchvision，再安装 `requirements-detection.txt`。当前本地无法访问上述 AutoDL 路径，需在数据所在服务器执行。

结构：`ResNeXt-101 → FPN（P2/P3/P4/P5）→ 上下文增强 → FCOS 分类、框回归、中心度预测`。

- P2–P5 的步长为 4、8、16、32，保留高分辨率特征用于小目标检测。
- 上下文增强复用 `DenseContrastModule` 和 `SELayer`，并增加残差通路保留原始外观信息；增强模块采用 GroupNorm，减少对训练批量大小的依赖。骨干仍使用原有 BatchNorm。
- 基于 [torchvision FCOS](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.detection.FCOS.html) 实现样本匹配、Focal/GIoU/中心度损失、解码与按类别 NMS。
- `UAVDetectionHead` 可以独立接收其他网络的多尺度特征，但不是可直接替换 YOLO Detect 的模块；接入其他框架需要适配特征、标签和损失接口。
- 这是可训练的结构改造，尚未进行恶劣天气数据训练或精度评测，也没有去雾/去雨监督模块。需要使用实际目标类别及对应边界框标注训练；不能直接使用 `GSD.pth` 得到有效检测结果。

### 环境与测试

检测依赖见 `requirements-detection.txt`。CPU 验证环境可按以下命令创建；GPU 训练应安装相匹配的 CUDA 版 PyTorch/torchvision。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-detection.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 每轮训练输出大、中、小目标指标

新增 `training.py`。使用 `fit()` 训练时，每个 epoch 训练结束后会在验证集上分别输出整体、大目标、中目标、小目标的 **AP@[IoU=0.50:0.95] 和 AR@100**，同时显示训练 loss、AP50 和 AP75。指标按整个验证集累计计算，不对批次 AP 求平均。

尺寸使用传入验证图片中的标注框面积 `(x2-x1)*(y2-y1)`，在模型内部缩放之前计算。采用 [COCO 官方评估实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) 的默认区间：small 为 `[0,32²]`，medium 为 `[32²,96²]`，large 为 `[96²,100000²]`，单位像素平方。严格沿用官方端点规则，恰好等于 32² 或 96² 的标注会计入相邻两档。这里使用框面积，不使用分割面积。

```python
import torch
from torch.utils.data import DataLoader
from detection import build_uav_detector
from training import detection_collate, fit

# train_dataset / val_dataset 由你的数据读取代码提供。
# 每个样本为 (RGB float [3,H,W], {"boxes": float [N,4], "labels": int64 [N]})。
# 验证集使用固定预处理，不使用随机裁剪/增强；类别编号从 0 开始。
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                          collate_fn=detection_collate, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                        collate_fn=detection_collate, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_uav_detector(num_classes=3).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
history = fit(model, train_loader, val_loader, optimizer,
              num_classes=3, epochs=50, device=device, output_dir="runs/exp1")
```

输出列为：

```text
Epoch ... | loss=... | AP50=... | AP75=...
目标尺度       AP@[.50:.95] (%)    AR@100 (%)
整体                  ...               ...
大目标                ...               ...
中目标                ...               ...
小目标                ...               ...
```

- 每轮记录写入 `runs/exp1/metrics.csv`，包含 `AP_small/medium/large`、`AR_small/medium/large` 等列。控制台显示百分数，CSV 保存 0–1 原始值。某尺度没有有效标注时显示 `N/A`，CSV 留空；有标注但没有检出时为 0。
- 最新模型和优化器状态保存为 `runs/exp1/last.pth`，模型参数在 checkpoint 的 `model` 字段。每次独立实验使用新的输出目录，已有日志不会被覆盖。
- 验证时临时将 FCOS 分数阈值设为 0.001，完成后恢复。评估保留 COCO 的 `maxDets=[1,10,100]`；密集 UAV 场景中，这一截断可能影响召回。模型自身的 `detections_per_img` 应至少为 100。
- 可独立调用 `evaluate_by_size(model, val_loader, num_classes, device)`；模型须提前放到相应设备。本训练循环面向单进程，尚未实现分布式指标汇总。
- 原 `data_loader.py` 是玻璃分割数据读取器，不能直接提供检测框标注；需接入你的 UAV 检测数据集。

### 单步训练和推理接口

```python
import torch
from detection import build_uav_detector

# 示例为 3 个前景类别，按自己的数据集修改；不额外添加背景类别。
net = build_uav_detector(num_classes=3)
images = [torch.rand(3, 640, 960)]  # RGB float，范围 [0,1]，无需预先归一化
targets = [{
    "boxes": torch.tensor([[100., 120., 140., 160.]]),  # 原图像素坐标 xyxy
    "labels": torch.tensor([0], dtype=torch.int64),    # 连续类别编号 0..2
}]

net.train()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()
losses = net(images, targets)
sum(losses.values()).backward()
optimizer.step()

# 实际推理前需要训练好的检测模型参数；此处仅演示 API。
net.eval()
with torch.no_grad():
    predictions = net(images)
# predictions[0]: boxes [N,4]、labels [N]、scores [N]，框已恢复到原图尺寸。
```

默认按短边 800、长边最多 1333 等比例缩放，可用 `min_size` / `max_size` 调整。批内图片自动填充到 32 的倍数。保存和加载完整检测权重使用 `net.state_dict()` / `net.load_state_dict(...)`。`backbone_path` 仅接受原始 ResNeXt 分类骨干权重文件，不接受 GlassNet 的 `GSD.pth`。不传权重时模型随机初始化，不会自动下载。

只使用检测头：

```python
from detection import UAVDetectionHead

head = UAVDetectionHead(in_channels=256, num_classes=3, num_levels=4)
# 每层形状 [B,256,H,W]，按高分辨率到低分辨率排列。
features = [torch.rand(1, 256, h, h) for h in (64, 32, 16, 8)]
raw = head(features)
```

`raw` 包含 `cls_logits [B,L,K]`、`bbox_regression [B,L,4]`、`bbox_ctrness [B,L,1]`，L 为各层位置数量总和。框为按步长归一化的左/上/右/下距离，分类与中心度为 logits；原始输出不能直接当作最终框使用。`build_uav_detector` 负责完整解码和后处理。没有目标的图片应提供形状为 `[0,4]` 的 boxes 和 `[0]` 的 int64 labels。

下面保留原始玻璃分割项目说明。

本项目包含论文 **Rich Context Aggregation with Reflection Prior for Glass Surface Detection**（CVPR 2021）的模型与推理代码。作者为 Jiaying Lin、Zebang He 和 Rynson W. H. Lau。原始说明见 [readme.txt](readme.txt)，项目主页：https://jiaying.link/cvpr2021-gsd/ 。

## 项目用途

输入 RGB 图片，逐像素预测玻璃区域，并输出三通道反射预测图。ResNeXt-101 32×4d 是特征提取骨干，完整任务是玻璃区域分割。

处理流程：图片缩放至 384×384 并归一化 → ResNeXt 多层特征提取 → 多尺度上下文差分与注意力 → 逐层预测 → RefNet 联合细化玻璃预测和反射预测 → CRF 后处理 → 保存至原图尺寸。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `model.py` | `GlassNet` 主网络；`DenseContrastModule` 对不同膨胀率的特征作差；`SELayer` 执行上下文及通道注意力；`RefNet` 输出玻璃预测和反射预测。 |
| `backbone/resnext/resnext_101_32x4d_.py` | ResNeXt-101 32×4d 网络定义。 |
| `backbone/resnext/resnext101_regular.py` | 将骨干划分为五级特征提取模块，支持加载骨干权重。 |
| `data_loader.py` | 图片、标签和反射图读取，以及缩放、裁剪、翻转、归一化和张量转换。 |
| `infer.py` | 加载训练好的模型，批量推理并保存分割图与反射图。 |
| `misc.py` | CRF 细化、IoU / MAE / BER 等评估函数和辅助工具。 |

`GlassNet.forward()` 返回五个单通道分割预测（logits）和一个三通道反射预测；推理脚本使用细化后的 `d0` 作为最终分割结果。

## 推理准备

当前代码导入的第三方依赖包括 `torch`、`torchvision`、`numpy`、`scikit-image`、`Pillow`、`pydensecrf` 和 `xlwt`。仓库没有提供已验证的依赖版本组合。

1. 准备与本机 CUDA 环境匹配的 PyTorch 及上述依赖。
2. 从原作者链接下载 [GSD.pth](https://drive.google.com/file/d/1SZgzvddRpa8BQf0hOf5_d1_ImcKso9yK/view?usp=sharing)，放在项目根目录。
3. 将测试图片放入 `GSD/test/image/`。
4. 在项目根目录执行 `python infer.py`。

默认输出：

- 玻璃区域预测：`released_gsd_results/GSD/`
- 反射预测：`released_gsd_reflections/GSD/`

## 当前代码的运行限制

- 仓库未附带数据集、模型权重、训练入口或完整评估入口；辅助函数不构成完整训练流程。
- `infer.py` 在启动时直接调用 `torch.cuda.set_device(0)`，当前实现要求可用的 CUDA GPU。
- 推理入口位于模块顶层，且 DataLoader 使用 `num_workers=1`；Windows 多进程运行时需要添加主入口保护，或调整为单进程加载。
- 路径和输入尺寸在脚本内固定；部分文件名处理使用 `/` 分隔符，Windows 路径可能需要适配。
- 代码包含 `F.upsample`、`F.sigmoid` 等旧式 API。未进行完整推理验证，不能保证与当前依赖版本直接兼容。

## 来源与许可

保留原始 [license.txt](license.txt) 和 [readme.txt](readme.txt)。使用和再分发应遵循原始许可文本；其中注明仅限非商业用途。

```bibtex
@inproceedings{GSD:2021,
    title={Rich Context Aggregation with Reflection Prior for Glass Surface Detection},
    author={Lin, Jiaying and He, Zebang and Lau, Rynson W.H.},
    booktitle={Proc. CVPR},
    year={2021}
}
```
