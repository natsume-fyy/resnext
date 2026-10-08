# 玻璃表面检测（Glass Surface Detection）

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
