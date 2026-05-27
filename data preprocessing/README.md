# CSI 数据预处理工具包

针对 ESP32-S3 四节点 CSI 采集系统的数据预处理代码，支持从原始 CSV 到 PyTorch DataLoader 的完整流程。

## 快速开始

### 1. 安装依赖

```bash
cd "data preprocessing"
pip install -r requirements.txt
```

### 2. 修改配置

编辑 `config.yaml`，确认数据路径：

```yaml
data:
  csv_paths:
    - "../CSI collection/data/*.csv"
```

### 3. 运行完整预处理

```bash
python example_train_ready.py
```

输出：
- `processed/train_windows.npz` — 训练集窗口
- `processed/val_windows.npz` — 验证集窗口（如有多 session）
- `processed/test_windows.npz` — 测试集窗口
- `processed/norm_stats.pkl` — 归一化统计量

### 4. 快速测试单文件

```bash
python example_load.py
```

---

## 文件结构

| 文件 | 说明 |
|---|---|
| `config.yaml` | 预处理超参数配置 |
| `preprocess.py` | 核心预处理：加载 → 过滤 → I/Q 转幅度 → 归一化 → 滑动窗口 |
| `dataset.py` | PyTorch `Dataset` 封装，支持数据增强和加权采样 |
| `augmentation.py` | 数据增强：时间偏移/拉伸、高斯噪声、Mixup、子载波 Mask |
| `segment.py` | 动作自动切分：基于方差阈值从完整片段中切出动作核心段 |
| `features.py` | 手工特征提取（供 XGBoost / Random Forest 使用） |
| `utils.py` | 工具函数：I/Q 解析、幅度/相位转换、归一化 |
| `example_load.py` | 单文件加载示例 |
| `example_train_ready.py` | 完整训练数据准备示例（划分 → 预处理 → Dataset → DataLoader） |

---

## 核心功能详解

### 异常帧过滤

自动丢弃以下异常帧：
- `first_word_invalid == 1` 的帧
- `csi_len != 128` 的帧
- I/Q 解析数量 ≠ 64 的帧
- RSSI 超出范围的帧（可选）

### I/Q 转换

```python
from utils import parse_iq_bytes, iq_to_amplitude

iq = parse_iq_bytes("0 0 -40 -52 ...")  # int8 [128]
amp = iq_to_amplitude(iq)                # float32 [64]
```

### 滑动窗口

```python
from preprocess import extract_windows

windows, labels = extract_windows(
    amp_matrix,      # [n_frames, n_subcarriers]
    labels_array,    # [n_frames]
    window_size=128, # 约 1.5s @87fps
    stride=64,       # 50% 重叠
    label_strategy="majority"
)
# windows: [n_windows, 128, 64]
```

### 数据增强

```python
from augmentation import augment_window

aug_window = augment_window(window, {
    "time_shift": True,
    "time_shift_range": 10,
    "gaussian_noise": 0.01,
    "amplitude_scale": 0.1,
})
```

### 动作自动切分

```python
from segment import segment_action

segments = segment_action(
    amp_matrix, labels,
    method="variance",
    baseline_window=50,
    threshold_ratio=3.0,
    min_segment_len=30,
    padding=15,
)
# 返回: [(sub_amp, sub_labels, start, end), ...]
```

### 手工特征（传统 ML）

```python
from features import extract_all_features

feat_vector = extract_all_features(
    window,           # [T, C]
    rssi_window,      # [T] 可选
    cfg={"temporal_stats": True, "diff_stats": True}
)
# feat_vector: [n_features] 一维向量，可直接送入 XGBoost
```

---

## 配置参数速查

### 窗口设置 (`window`)

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `size` | 128 | 窗口帧数（128/87fps ≈ 1.5s） |
| `stride` | 64 | 步长，50% 重叠 |
| `label_strategy` | majority | majority / center / last |

### 归一化 (`preprocess.normalize`)

| 方法 | 适用场景 |
|---|---|
| `zscore` | 推荐，稳定 |
| `minmax` | 需固定范围 [0,1] 时 |
| `energy` | 消除不同距离带来的幅度差异 |
| `none` | 不做归一化 |

### 数据增强 (`augment`)

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `gaussian_noise` | 0.01 | 噪声强度 = 0.01 × std |
| `amplitude_scale` | 0.1 | 幅度缩放范围 ±10% |
| `subcarrier_mask_ratio` | 0.05 | 随机屏蔽 5% 子载波 |
| `mixup_alpha` | 0.4 | Mixup Beta 分布参数 |

---

## 使用示例：在训练脚本中加载

```python
import numpy as np
from dataset import CSIDataset, load_processed_npz

# 加载预处理后的数据
windows, labels = load_processed_npz("processed/train_windows.npz")

# 定义标签映射
label_map = {"idle": 0, "walking": 1, "sitting_down": 2, "standing_up": 3, "fall": 4}

# 创建 Dataset
train_ds = CSIDataset(windows, labels, label_map, is_training=True)

# 创建 DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 训练循环
for batch_x, batch_y in train_loader:
    # batch_x: [32, 128, 64]
    # batch_y: [32]
    ...
```

---

## 注意事项

1. **Session 划分**：强烈建议 `split_by_session: true`，即按采集日期划分训练/验证/测试集。这比随机划分更能验证模型的跨时间泛化能力。

2. **去零子载波**：HT-LTF 的 64 个子载波中，守卫带（guard subcarriers）始终为 0。预处理会自动去掉这些子载波，通常保留 52-56 个有效子载波。

3. **idle 数据质量**：如果 idle 数据的 RSSI 出现剧烈双模态跳变（如 -16 dBm 与 -29 dBm 交替），说明存在多个 ping 源。请先修复固件配置，确保 CSI 只来自单一发射源。

4. **动作切分**：如果采集时记录了"静止 → 动作 → 静止"的完整片段，建议开启 `segment.enabled: true` 自动切分出动作核心段。

5. **类别不平衡**：`fall` 样本通常最少。建议设置 `class_weights.fall: 3.0` 并开启 `oversample: true`。
