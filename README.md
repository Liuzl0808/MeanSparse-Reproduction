# MeanSparse 复现代码（CIFAR-10 / SmallCNN）

课程论文《面向图像分类器的对抗性攻击与鲁棒防御方法研究》的复现代码与实验数据。

- **复现对象**：MeanSparse: Post-Training Robustness Enhancement through Mean-Centered Feature Sparsification（Amini et al., ICLR 2025，arXiv:2406.05927）
- **原始实现**：https://github.com/SPIN-UMass/MeanSparse
- **本文工作**：在官方实现基础上做适配——改用 forward hook 把稀疏化算子挂到 SmallCNN 每个 ReLU 之前（Conv → BN → MS → ReLU），并在 CIFAR-10 训练集上重新统计各通道的均值与标准差。方法本身并非本文提出，本仓库只完成复现与评测。

## 环境

Python 3.10+ / PyTorch 2.11.0，单张消费级显卡即可。
CIFAR-10 数据集在首次运行时自动下载到本地 `data/`，不包含在仓库中。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `train2_adversarial_training.py` | PGD-7 对抗训练 + 权重平均，产出 AT / AT+SWA 两个检查点 |
| `mean_sparse_repro.py` | MeanSparse 复现：统计通道统计量、α 网格搜索、完整评测 |
| `plot_alpha_search.py` | 画 α 搜索曲线（论文图 3） |
| `demo1_adversarial_examples.py` | 生成对抗样本示例图（论文图 2） |
| `alpha_search.json` | α 从 0.05 到 0.50 的搜索原始记录 |
| `table_ms.csv` | 表 1 后两行（AT+MeanSparse / AT+SWA+MeanSparse） |
| `table1_accuracy.csv` | 表 1 第 1 行（标准训练） |
| `table2_3.csv` | 表 1 中间两行（PGD-AT / AT+SWA） |

## 复现步骤

```bash
python train2_adversarial_training.py --epochs 12   # 产出 model_at.pt、model_atswa.pt
python mean_sparse_repro.py                          # 产出 alpha_search.json、table_ms.csv
python plot_alpha_search.py                          # 产出 fig3_alpha_search.pdf
```

## 主要结果

攻击设置：ℓ∞ 约束，ε = 8/255；PGD 评测步数 10，步长 2/255。

| 方法 | 干净准确率(%) | FGSM(%) | PGD-10(%) |
| --- | --- | ---: | ---: |
| 标准训练 | 83.01 | 4.32 | 0.00 |
| PGD-AT | 70.07 | 42.99 | 39.08 |
| AT+SWA | 67.07 | 44.01 | 41.13 |
| AT+MeanSparse (α=0.45) | 68.89 | 46.46 | 43.42 |
| AT+SWA+MeanSparse (α=0.45) | 66.70 | 46.42 | 43.81 |

## 说明

实验中的模型权重文件（约 8.8 MB/个）因体积原因未上传，运行上述脚本可在本地重新生成。
