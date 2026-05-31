# CSI 数据预处理工具包

面向 ESP32-S3 CSI 采集 CSV 的训练数据准备工具。它把原始串口 CSV 转成稳定的窗口数据，并可直接创建 PyTorch `Dataset` / `DataLoader`。

当前主流程支持：

- 异常帧过滤：`first_word_invalid`、`csi_len`、I/Q 数量、RSSI 范围
- I/Q 特征：`amplitude`、`phase`、`both`
- 训练集拟合并复用去零子载波索引，避免 train/val/test 维度漂移
- 归一化：`zscore`、`minmax`、`energy`、`none`
- 可选动作自动切分/重标注
- 滑动窗口切分和窗口级元数据保存
- 数据增强、类别加权采样、Mixup collator
- 手工特征提取，供 XGBoost / Random Forest 使用

## 快速开始

### 1. 安装依赖

```bash
cd "data preprocessing"
python3 -m pip install -r requirements.txt
```

### 2. 确认数据路径

默认配置读取当前项目里的 AP/STA 采集数据：

```yaml
data:
  csv_paths:
    - "../CSI_collection_AP_STA/data/*.csv"
```

如果你的 CSV 放在其他目录，改 `config.yaml` 里的 `data.csv_paths`。

### 3. 快速测试单文件

```bash
python3 example_load.py
```

它会读取第一个匹配到的 CSV，输出窗口数量、窗口形状和标签集合，并保存 `processed/demo_windows.npz`。

### 4. 生成训练/验证/测试数据

```bash
python3 example_train_ready.py
```

输出文件：

- `processed/train_windows.npz`
- `processed/val_windows.npz`，有验证 split 时生成
- `processed/test_windows.npz`，有测试 split 时生成
- `processed/norm_stats.pkl`

`norm_stats.pkl` 保存训练集拟合出来的归一化统计量、`keep_idx` 和 `output_type`。验证集、测试集和实时推理必须复用它。

如果当前环境没有安装 PyTorch，`example_train_ready.py` 仍会完成 `.npz` 预处理保存，只会跳过 `Dataset` / `DataLoader` 演示。

## 数据划分

默认使用和 Baseline 一致的 `split_strategy: stratified_label`，脚本会从文件名解析动作标签，并在每个类别内部按比例切分 train/val/test。

```yaml
data:
  split_ratio:
    train: 0.7
    val: 0.15
    test: 0.15
  split_strategy: "stratified_label"
```

可选划分策略：

| 策略 | 说明 |
| ---- | ---- |
| `stratified_label` | 从文件名解析动作标签，每个类别内部按比例切 train/val/test，推荐快速训练/调试 |
| `session` | 按文件名里的 `YYYYMMDD` 分 session，推荐跨 session 泛化评估 |
| `file` | 按排序后的文件列表直接划分 |

`stratified_label` 需要文件名类似：

```text
node1_idle_20260528_230438.csv
node1_walking_20260528_234548.csv
node1_sitting_down_20260529_000144.csv
```

注意：

- 同一个 CSV 不会同时进入 train 和 val/test。
- 如果用 `session` 且只有 1 个 session，脚本只生成训练集，并提示没有验证集。
- 如果用 `session` 且所有动作都在同一天采集，所有动作都会落在同一个 split。快速验证时建议先用 `stratified_label`。

## 核心配置

### I/Q 输出

```yaml
preprocess:
  output_type: "amplitude"  # amplitude | phase | both
```

- `amplitude`：推荐首选，输出 `[T, C]`
- `phase`：相位容易受频偏影响，建议作为对比实验
- `both`：输出 `[T, 2*C]`，幅度和相位拼接

### 去零子载波

```yaml
preprocess:
  remove_zero_subcarriers: true
```

训练集会拟合 `keep_idx`，验证/测试复用同一组索引。不要让每个 split 各自计算，否则窗口维度可能不一致。

### 归一化

```yaml
preprocess:
  normalize: "zscore"  # none | minmax | zscore | energy
```

推荐先用 `zscore`。`energy` 会对每帧做 L2 归一化，适合想削弱距离/整体增益影响的实验。

### 滑动窗口

```yaml
window:
  size: 128
  stride: 64
  min_windows: 2
  label_strategy: "majority"  # majority | center | last
```

`min_windows` 会过滤窗口数太少的文件。窗口有重叠时，不要随机打散后再划分 train/val，否则评估会虚高。

### 动作自动切分

```yaml
segment:
  enabled: false
  method: "diff_energy"      # variance | diff_energy
  baseline_window: 50
  smooth_window: 10
  threshold_ratio: 3.0
  min_segment_len: 30
  padding: 15
  default_label: "idle"
```

如果单个 CSV 结构是“静止 -> 动作 -> 静止”，可以开启切分。推荐先用 `diff_energy`，它比单帧子载波方差更贴近动作变化。

## 数据增强

训练时由 `CSIDataset` 调用 `augment_window`：

```yaml
augment:
  enabled: true
  time_shift: true
  time_stretch: true
  gaussian_noise: 0.01
  amplitude_scale: 0.1
  subcarrier_mask_ratio: 0.05
  timestep_mask_ratio: 0.05
  oversample: true
  class_weights:
    idle: 1.0
    walking: 1.0
    sitting_down: 1.2
    standing_up: 1.2
    fall: 3.0
```

`time_shift` 使用边缘帧填充，不做循环移位。`oversample: true` 时，`example_train_ready.py` 会启用 `WeightedRandomSampler`。

### Mixup

```yaml
augment:
  mixup: true
  mixup_alpha: 0.4
```

开启后，训练 DataLoader 会使用 `MixupCollator`，标签形状从 `[B]` 变成 soft one-hot `[B, num_classes]`。训练时需使用支持 soft label 的损失，例如：

```python
import torch.nn.functional as F

def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
```

如果你的训练脚本只支持 `nn.CrossEntropyLoss` 的整数标签，先把 `mixup: false`。

## 在训练脚本中加载

```python
from dataset import CSIDataset, load_processed_split
from torch.utils.data import DataLoader

windows, labels = load_processed_split("processed/train_windows.npz")
label_map = {
    "idle": 0,
    "walking": 1,
    "sitting_down": 2,
    "standing_up": 3,
    "fall": 4,
}

train_ds = CSIDataset(
    windows=windows,
    labels=labels,
    label_map=label_map,
    augment_cfg=None,
    is_training=True,
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

for batch_x, batch_y in train_loader:
    # batch_x: [B, T, C]
    # batch_y: [B]
    ...
```

2D-CNN 可使用 `collate_fn_2d` 得到 `[B, 1, T, C]`；TCN 可使用 `collate_fn_tcn` 得到 `[B, C, T]`。

## 手工特征

```python
from features import extract_all_features

feat = extract_all_features(
    window,
    rssi_window=None,
    cfg={
        "temporal_stats": ["mean", "std", "max", "min", "ptp", "energy"],
        "diff_stats": ["mean", "std", "energy"],
        "inter_carrier_corr": True,
    },
)
```

`load_processed_split` 也支持大数据推荐格式：`train_windows.npy` + 同目录 `train_labels.npy`。`.npy` 会用 mmap 懒加载，避免训练开始时一次性把全部窗口读进内存。

配置里的 `temporal_stats` 和 `diff_stats` 可以直接写列表；也可以设为 `false` 禁用对应特征。

## 文件说明

| 文件                       | 说明                                                           |
| -------------------------- | -------------------------------------------------------------- |
| `config.yaml`            | 预处理、窗口、切分、增强配置                                   |
| `preprocess.py`          | 核心 pipeline：CSV -> 特征矩阵 -> 固定子载波 -> 归一化 -> 窗口 |
| `dataset.py`             | PyTorch `Dataset`、加权采样器、2D/TCN collate                |
| `augmentation.py`        | 时间/幅度增强和 Mixup collator                                 |
| `segment.py`             | 动作切分和重标注                                               |
| `features.py`            | 传统 ML 手工特征                                               |
| `utils.py`               | CSV、I/Q、归一化工具函数                                       |
| `example_load.py`        | 单文件 smoke test                                              |
| `example_train_ready.py` | 完整训练数据准备示例                                           |

## 实验注意事项

1. 按 session/录制轮次划分后再切窗口，避免重叠窗口泄漏。
2. `norm_stats.pkl` 是训练和部署的一部分，实时推理必须复用。
3. 当前工具处理的是单节点文件。四节点 Early Fusion 需要额外做时间对齐，再把节点维度拼接。
4. `fall` 样本通常少，除了类别权重，最好增加独立采集次数。
5. 先跑传统 ML baseline 验证数据质量，再训练 CNN/TCN/LSTM。
