# CSI 人体瞬时动作识别网络设计方案

> 针对 ESP32-S3 四节点 CSI 采集系统，设计从数据预处理到模型部署的完整方案。
> 目标类别：`idle` | `walking` | `sitting_down` | `standing_up` | `fall`

---

## 1. 数据特性分析

### 1.1 原始数据规格

| 参数 | 数值 | 说明 |
|---|---|---|
| 芯片 | ESP32-S3 | 单天线 |
| CSI 长度 | 128 bytes | HT-LTF，64 个 I/Q 对 |
| 带宽 | 20 MHz | `cwb=0`，`secondary_channel=0` |
| 帧率 | ~87 fps | 固件内自 ping，间隔约 11.5 ms |
| 子载波数 | 64 | 含 DC 和保护间隔子载波 |
| 有效子载波 | ~52-56 | 去掉始终为 0 的守卫带 |
| 节点数 | 4 | 不同空间位置 |
| 采集标签 | 5 类 | idle / walking / sitting_down / standing_up / fall |

### 1.2 关键字段说明

当前 CSV 已包含以下改进字段（相比最初版本）：

- `src_mac` / `dst_mac`：MAC 过滤后的源/目的地址，确保数据来自目标链路
- `first_word_invalid`：CSI 首字有效性标记，`1` 表示该帧建议丢弃
- `rx_seq`：802.11 MAC 层序列号，可用于丢包检测
- `local_time_us` / `rx_timestamp_us`：发送与接收时间戳，可算往返时延

### 1.3 数据质量检查清单

每批数据采集后建议运行：

```bash
python3 tools/inspect_csi_settings.py data/xxx.csv
```

需满足：
- `I/Q pairs parsed = 64` 占比 > 95%
- `first_word_invalid = 0` 占比 > 95%
- `Frame interval` 均值在 8-20 ms 之间（帧率 50-120 fps）
- `RSSI` 均值在 -35 ~ -65 dBm 之间

---

## 2. 数据预处理与特征工程

### 2.1 预处理流水线

```
原始 CSV
  │
  ▼
[Step 1] 过滤异常帧
  ├── 丢弃 first_word_invalid == 1 的帧
  ├── 丢弃 csi_len != 128 的帧
  └── 丢弃 I/Q 解析数量 != 64 的帧
  │
  ▼
[Step 2] I/Q → 幅度 / 相位
  ├── amplitude = sqrt(I² + Q²)        # 主特征，建议先用幅度
  └── phase   = atan2(Q, I)            # 可选，易受频偏影响
  │
  ▼
[Step 3] 去零子载波
  └── 去掉在所有帧中幅度恒为 0 的子载波索引
  │
  ▼
[Step 4] 滑动窗口切分
  ├── 窗长：128 帧（约 1.5s）或 256 帧（约 3s）
  ├── 步长：50% 重叠（64 帧或 128 帧）
  └── 标签：窗口内出现频率最高的标签（或取中心帧标签）
  │
  ▼
[Step 5] 归一化
  ├── 方案 A：Min-Max 归一化到 [0, 1]（按全局统计量）
  ├── 方案 B：Z-score 标准化（按全局均值方差）
  └── 方案 C：每帧能量归一化（除以该帧幅度平方和）
  │
  ▼
[Step 6] 四节点对齐（可选）
  └── 按 pc_time_iso 或 local_time_us 做最近邻对齐
```

### 2.2 手工特征（供传统 ML 方案使用）

对每个滑动窗口的每个子载波序列提取：

**时域统计（64 维 × 特征数）：**
- 均值（Mean）、标准差（Std）、最大值（Max）、最小值（Min）
- 极差（Peak-to-Peak）、能量（Energy = Σ|amp|²）
- 偏度（Skewness）、峰度（Kurtosis）
- 过零率（Zero Crossing Rate，对差分序列）

**子载波间特征：**
- 相邻子载波相关系数均值
- 全子载波协方差矩阵的 Frobenius 范数

**时序动态特征：**
- 相邻帧差分的均值、方差（反映变化剧烈程度）
- 一阶差分能量：Σ|amp[t] - amp[t-1]|²

**RSSI 特征：**
- RSSI 均值、方差、极差（反映整体链路变化）

四节点融合时，将上述特征按 `[node1_feats, node2_feats, node3_feats, node4_feats]` 拼接。

---

## 3. 动作切分与标注策略

### 3.1 问题

采集时通常记录的是完整片段：

```
[静止 2s] → [动作 3s] → [静止 2s]
```

如果直接把整段送入训练，模型会学到大量"静止"前缀/后缀的干扰模式。

### 3.2 自动切分算法（基于方差阈值）

```python
def segment_action(amp_matrix, rssi, window_size=50, threshold_ratio=3.0):
    """
    amp_matrix: [T, N_subcarriers] 幅度矩阵
    返回: 动作起止索引列表 [(start, end), ...]
    """
    # 1. 计算每帧全子载波幅度方差（或差分能量）
    frame_var = np.var(amp_matrix, axis=1)
    
    # 2. 滑动平均平滑
    smooth_var = np.convolve(frame_var, np.ones(10)/10, mode='same')
    
    # 3. 取前 1 秒作为基线
    baseline = np.mean(smooth_var[:window_size])
    threshold = baseline * threshold_ratio
    
    # 4. 检测超过阈值的连续区域
    above = smooth_var > threshold
    # ... 找连续 True 段，过滤过短片段
    
    return segments
```

### 3.3 标注规范

| 动作 | 建议时长 | 每次录制结构 | 窗口标签策略 |
|---|---|---|---|
| `idle` | 5-10 s | 全程静止 | 全部标记 idle |
| `walking` | 3-5 s | 静止→走→静止 | 仅中间走段标记 walking |
| `sitting_down` | 2-4 s | 站→坐→静止 | 仅坐下过程标记 sitting_down |
| `standing_up` | 2-4 s | 坐→站→静止 | 仅起立过程标记 standing_up |
| `fall` | 2-4 s | 站→倒→躺 | 仅跌倒过程标记 fall |

**建议：** 训练时只用切分后的核心动作窗口；`idle` 可以用完整片段。

---

## 4. 网络架构方案

### 方案 A：传统机器学习 Baseline（XGBoost / LightGBM）

**定位：** 快速验证数据可用性，强 Baseline。

**输入：** 每窗口手工特征向量，维度约 200-400 维（单节点）或 800-1600 维（四节点拼接）。

**模型：**

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=5,
    eval_metric='mlogloss'
)
```

**优缺点：**
- ✅ 训练秒级完成，不需要 GPU
- ✅ 小数据量（每类 50+ 样本）即可 work
- ✅ 可输出特征重要性，帮助分析哪些子载波/节点最敏感
- ❌ 无法捕捉精细时序模式，对相似动作（sit/stand）区分力有限

**预期准确率：** 70-85%（四节点融合后）

---

### 方案 B：1D-CNN + BiLSTM（⭐ 推荐主方案）

**定位：** 兼顾空间子载波特征与时序动态，学术与工程中最常用架构。

**输入张量：** `[batch_size, time_steps, n_subcarriers]` = `[N, 128, 64]`

> `time_steps = 128` 对应约 1.5 秒窗口（按 87 fps 计）。对 fall 等短动作可减小到 64；对 walking 等可增大到 256。

**网络结构：**

```
Input: [N, 128, 64]
│
├─ Conv1D(in=64, out=32, kernel=5, padding=2)
├─ BatchNorm1D(32)
├─ ReLU
├─ Conv1D(in=32, out=32, kernel=5, padding=2)
├─ BatchNorm1D(32)
├─ ReLU
├─ MaxPool1D(kernel=2)          → [N, 64, 32]
│
├─ Conv1D(in=32, out=64, kernel=3, padding=1)
├─ BatchNorm1D(64)
├─ ReLU
├─ MaxPool1D(kernel=2)          → [N, 32, 64]
│
├─ Conv1D(in=64, out=128, kernel=3, padding=1)
├─ BatchNorm1D(128)
├─ ReLU
├─ MaxPool1D(kernel=2)          → [N, 16, 128]
│
├─ Permute / Reshape 为 [N, 16, 128]
├─ BiLSTM(input=128, hidden=64, num_layers=2, dropout=0.3)
│   └─ 输出: [N, 16, 128] (双向拼接)
│
├─ Self-Attention (optional)
│   ├─ Q = K = V = LSTM 输出
│   └─ 输出: [N, 16, 128]
│
├─ GlobalAvgPool1D (沿时间轴)   → [N, 128]
│   └─ 或取 LSTM 最后一个时间步
│
├─ Dropout(0.5)
├─ Linear(128 → 64) + ReLU
├─ Dropout(0.3)
└─ Linear(64 → 5)              → 5 类 logits
```

**PyTorch 代码骨架：**

```python
import torch
import torch.nn as nn

class CNN_BiLSTM(nn.Module):
    def __init__(self, n_subcarriers=64, n_classes=5, time_steps=128):
        super().__init__()
        
        # 1D-CNN: 沿子载波维度卷积，提取局部频域模式
        self.cnn = nn.Sequential(
            nn.Conv1d(n_subcarriers, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 -> 64
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 64 -> 32
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 32 -> 16
        )
        
        # BiLSTM: 沿时间维度建模
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=64,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),  # BiLSTM: 64*2=128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        # x: [N, T, C] = [N, 128, 64]
        x = x.permute(0, 2, 1)   # [N, 64, 128] for Conv1d
        x = self.cnn(x)          # [N, 128, 16]
        x = x.permute(0, 2, 1)   # [N, 16, 128] for LSTM
        x, _ = self.lstm(x)      # [N, 16, 128]
        x = torch.mean(x, dim=1) # GlobalAvgPool -> [N, 128]
        return self.classifier(x)
```

**优缺点：**
- ✅ 同时捕捉子载波局部模式（CNN）和动作时间演变（LSTM）
- ✅ 在 CSI 动作识别论文中验证有效
- ✅ 对 sit/stand 等相似动作区分力强
- ❌ 需要 GPU 训练，参数量中等（~200K-500K）
- ❌ 需要一定数据量（建议每类 200+ 窗口）

**预期准确率：** 85-95%（四节点融合后）

---

### 方案 C：2D-CNN 时空卷积

**定位：** 将 CSI 矩阵视为"图像"，用 2D 卷积同时提取时间和子载波维度的局部特征。

**输入张量：** `[batch_size, 1, time_steps, n_subcarriers]` = `[N, 1, 128, 64]`

**网络结构（轻量版）：**

```
Input: [N, 1, 128, 64]
│
├─ Conv2d(1→16, kernel=3×3) + BN + ReLU
├─ Conv2d(16→16, kernel=3×3) + BN + ReLU
├─ MaxPool2d(2×2)              → [N, 16, 64, 32]
│
├─ Conv2d(16→32, kernel=3×3) + BN + ReLU
├─ MaxPool2d(2×2)              → [N, 32, 32, 16]
│
├─ Conv2d(32→64, kernel=3×3) + BN + ReLU
├─ AdaptiveAvgPool2d(1×1)      → [N, 64, 1, 1]
│
├─ Flatten                     → [N, 64]
├─ Dropout(0.5)
└─ Linear(64 → 5)
```

**优缺点：**
- ✅ 概念直观，类似处理语谱图
- ✅ 可尝试预训练骨干（如 EfficientNet 微调到 128×64 分辨率）
- ❌ 对小型数据集容易过拟合
- ❌ 时间维度的长程依赖不如 LSTM/Transformer

**预期准确率：** 80-92%

---

### 方案 D：Temporal Convolutional Network (TCN)

**定位：** 用纯卷积替代 RNN，解决 LSTM 训练慢、难并行的问题。

**核心思想：**
- 因果卷积（Causal Convolution）：确保预测第 t 帧只用 0~t 信息
- 空洞卷积（Dilated Convolution）：指数级扩大感受野
- 残差连接：缓解梯度问题

**输入张量：** `[N, n_subcarriers, time_steps]` = `[N, 64, 128]`（ channels-first ）

**网络结构：**

```python
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        
        # 因果：截断末尾 pad 长度
        out = out[:, :, :x.size(2)]
        
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)

class TCN(nn.Module):
    def __init__(self, n_subcarriers=64, n_classes=5):
        super().__init__()
        layers = []
        dilations = [1, 2, 4, 8]
        channels = [64, 64, 128, 128]
        
        in_ch = n_subcarriers
        for out_ch, dil in zip(channels, dilations):
            layers.append(TemporalBlock(in_ch, out_ch, dilation=dil))
            in_ch = out_ch
        
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], n_classes)
    
    def forward(self, x):
        # x: [N, T, C] -> [N, C, T]
        x = x.permute(0, 2, 1)
        x = self.tcn(x)      # [N, 128, T]
        x = self.gap(x).squeeze(-1)  # [N, 128]
        return self.fc(x)
```

**优缺点：**
- ✅ 训练速度快（卷积可完全并行）
- ✅ 感受野随层数指数增长，适合长序列
- ✅ 无梯度消失问题
- ❌ 调参比 LSTM 稍复杂（空洞率设计）

**预期准确率：** 83-93%

---

### 方案 E：轻量级边缘部署网络（TinyCNN / Micro-MLP）

**定位：** 最终要在 ESP32-S3 上实时推理，或每个节点只发少量特征到 PC。

**策略：**
1. **节点端：** 每个 ESP32-S3 本地维护一个 128 帧滑动窗口，提取轻量特征（均值、方差、差分能量、子载波相关系数），约 20-30 维
2. **传输：** 通过 UDP 每秒发送 1-5 次特征向量到 PC/树莓派
3. **推理端：** PC 上运行一个小型 MLP（2-3 层，共几千参数）做四节点融合分类

**MLP 结构：**

```python
class TinyMLP(nn.Module):
    def __init__(self, in_dim=120, n_classes=5):  # 4 nodes × 30 feats
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)
```

**若要在 ESP32-S3 本地直接推理：**
- 使用 `esp-tflite-micro` 或 `esp-nn` 加速库
- 将 PyTorch 模型转为 TFLite Micro 格式
- 输入：当前窗口的 64 维均值 + 64 维方差 = 128 维
- 模型大小需控制在 < 50 KB

**优缺点：**
- ✅ 无线缆束缚，四节点可电池供电
- ✅ 延迟低（本地推理 < 50 ms）
- ❌ 准确率通常比深度学习方案低 5-10%
- ❌ 需要额外开发固件和通信协议

---

## 5. 四节点融合策略

### 5.1 Early Fusion（早期融合）⭐ 推荐

将四节点原始幅度矩阵直接拼接为 `[N, T, 4×64]` 或 `[N, 4, T, 64]`，输入单一网络。

```python
# 假设四节点已按时间对齐
node1 = ...  # [N, T, 64]
node2 = ...  # [N, T, 64]
node3 = ...  # [N, T, 64]
node4 = ...  # [N, T, 64]

# 方案 1：通道拼接
x = torch.cat([node1, node2, node3, node4], dim=-1)  # [N, T, 256]

# 方案 2：作为额外维度（适合 2D-CNN）
x = torch.stack([node1, node2, node3, node4], dim=1)  # [N, 4, T, 64]
# 然后 Conv2d(4→16, kernel=3×3)
```

**适用：** CNN_BiLSTM、TCN、2D-CNN

### 5.2 Late Fusion（晚期融合 / 决策级融合）

每个节点独立训练一个单节点模型，推理时四节点分别输出概率，再投票或加权平均。

```python
# 训练阶段：每个节点独立训练相同架构的网络
# 推理阶段：
probs = []
for node in [node1, node2, node3, node4]:
    p = F.softmax(model(node), dim=-1)  # [N, 5]
    probs.append(p)

# 平均融合
avg_prob = torch.mean(torch.stack(probs), dim=0)  # [N, 5]
pred = torch.argmax(avg_prob, dim=-1)

# 或加权融合（按节点位置或信号质量赋权）
weights = [0.3, 0.2, 0.3, 0.2]  # 根据验证集学习
weighted_prob = sum(w * p for w, p in zip(weights, probs))
```

**适用：** 节点间时间不同步、或需要热插拔某节点时。

### 5.3 Intermediate Fusion（中间融合）

每个节点先过 CNN 提取局部特征，再在高层拼接。

```python
# 各节点独立 CNN 编码
cnn = nn.Conv1d(64, 32, kernel_size=5)
feat1 = cnn(node1)  # [N, T', 32]
feat2 = cnn(node2)
feat3 = cnn(node3)
feat4 = cnn(node4)

# 高层拼接后统一分类
fused = torch.cat([feat1, feat2, feat3, feat4], dim=-1)  # [N, T', 128]
```

**适用：** 节点数可变、希望学习节点间交互关系。

**建议：** 比赛/项目初期先用 **Early Fusion**，简单且效果通常最好。如果时间允许，对比 Late Fusion。

---

## 6. 数据增强策略

CSI 数据采集成本高，增强对提升泛化非常关键。

### 6.1 时间域增强

| 方法 | 操作 | 强度 |
|---|---|---|
| 时间偏移 | 窗口起始点随机偏移 ±10 帧 | 必用 |
| 时间拉伸 | 线性插值将窗口缩放到 0.8x ~ 1.2x 长度 | 推荐 |
| 随机裁剪 | 从长片段中随机取不同子窗口 | 必用 |
| Mixup | 两个窗口按 λ 混合，标签也混合 | 推荐 |

```python
# Mixup 示例
def mixup(x1, x2, y1, y2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
```

### 6.2 幅度域增强

| 方法 | 操作 | 适用场景 |
|---|---|---|
| 高斯噪声 | `amp += N(0, 0.01 * std)` | 通用 |
| 幅度缩放 | `amp *= uniform(0.9, 1.1)` | 模拟不同距离 |
| 子载波 Mask | 随机将 5-10% 子载波置 0 | 模拟子载波丢失 |
| 时间步 Mask | 随机将 5% 时间步置 0 | 模拟丢包 |

### 6.3 节点级增强

| 方法 | 操作 |
|---|---|
| 节点 dropout | 训练时随机丢弃 1 个节点的输入（置 0） |
| 节点置换 | 改变四节点的空间顺序（如果位置不对称） |

### 6.4 类别不平衡处理

`fall` 样本通常最少，且最重要：

1. **过采样（Oversampling）：** `fall` 类在 DataLoader 中采样权重 ×3-5
2. **Focal Loss：** 降低易分类样本（如 `idle`）的损失权重
3. **类别权重：**

```python
class_weights = torch.tensor([1.0, 1.0, 1.2, 1.2, 2.5])  # fall 权重最高
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## 7. 训练流程与超参数

### 7.1 数据集划分

**强烈建议按 session / 日期划分，而非随机划分。**

```
Session 1 (Day 1):  训练集 60%
Session 2 (Day 2):  验证集 20%  ← 不同时间、不同位置
Session 3 (Day 3):  测试集 20%  ← 完全未见过
```

如果数据量小，用 **Leave-One-Session-Out (LOSO)** 交叉验证。

### 7.2 训练配置（以 CNN_BiLSTM 为例）

| 参数 | 设置 |
|---|---|
| 优化器 | AdamW |
| 初始学习率 | 1e-3 |
| Batch size | 32-64 |
| 学习率衰减 | CosineAnnealingLR(T_max=100) 或 ReduceLROnPlateau |
| 权重衰减 | 1e-4 |
| Dropout | 0.3-0.5 |
| Epoch | 100-200（EarlyStopping，耐心 15-20） |
| 损失函数 | CrossEntropyLoss（加权）或 FocalLoss |
| 评估频率 | 每 epoch 验证一次 |

### 7.3 Early Stopping 与 Checkpoint

```python
# 保存验证集 F1-macro 最高的模型
best_f1 = 0
for epoch in range(max_epochs):
    train(...)
    val_f1 = validate(...)
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pth')
        patience = 0
    else:
        patience += 1
        if patience > 20:
            break
```

---

## 8. 评估指标与测试方案

### 8.1 核心指标

| 指标 | 说明 | 目标值 |
|---|---|---|
| Accuracy | 整体准确率 | > 90% |
| Macro-F1 | 每类 F1 的均值（不受类别不平衡影响）| > 88% |
| `fall` Recall | 跌倒检出率（漏检代价高）| > 95% |
| `fall` Precision | 跌倒误报率 | > 85% |
| Inference Time | 单窗口推理耗时（GPU/CPU）| < 10 ms |

### 8.2 混淆矩阵分析

重点关注：
- `sitting_down` vs `standing_up`：两者易混淆，需看是否可通过时序特征区分
- `walking` vs `idle`：通常区分度最高
- `fall` vs 其他：fall 的特征通常是短时剧烈变化，如果模型把 fall 判为 walking，说明窗口太长或缺乏动态特征

### 8.3 消融实验建议

按以下顺序做消融，验证各模块贡献：

1. **Baseline：** 单节点 + 手工特征 + XGBoost
2. **+ 多节点：** 四节点拼接
3. **+ 深度学习：** CNN_BiLSTM 替代 XGBoost
4. **+ 数据增强：** 观察过拟合是否缓解
5. **+ 注意力：** Self-Attention 模块
6. **+ 切分优化：** 自动切分 vs 手动切分

---

## 9. 实施路线图

建议按以下顺序实施，每个阶段 1-3 天：

### Phase 1：数据就绪（第 1-2 天）

- [ ] 固件确认：MAC 过滤正常、帧率稳定、无大量异常帧
- [ ] 采集 idle 20 段、walking 20 段（单节点即可）
- [ ] 写数据加载器 `dataset.py`：CSV → 幅度矩阵 → 滑动窗口
- [ ] 用 `visualize_csi.py` 确认动作片段在幅度热图上有肉眼可见差异

### Phase 2：Baseline 验证（第 3-4 天）

- [ ] 实现手工特征提取
- [ ] 训练 XGBoost / Random Forest
- [ ] 若单节点准确率 > 70%，说明数据可用，继续；否则检查采集流程

### Phase 3：深度学习模型（第 5-8 天）

- [ ] 实现 CNN_BiLSTM（方案 B）
- [ ] 单节点实验 → 四节点 Early Fusion
- [ ] 调参：窗口大小、学习率、Dropout
- [ ] 加入数据增强

### Phase 4：完整采集与训练（第 9-12 天）

- [ ] 按 README 建议量采集全部 5 类数据
- [ ] 多 session 采集（至少 2-3 天不同时间）
- [ ] 自动切分算法优化
- [ ] 完整训练 + LOSO 交叉验证

### Phase 5：部署与实时测试（第 13-15 天）

- [ ] 导出 ONNX / TorchScript 模型
- [ ] 写实时推理脚本：读取串口 → 滑动窗口 → 模型推理 → 输出动作标签
- [ ] 四节点同时采集，PC 端融合推理
- [ ] 现场演示与调优

---

## 10. 代码文件规划

建议在工程根目录下新建 `model/` 文件夹：

```
CSI collection/
├── model/
│   ├── dataset.py          # CSI 数据加载与预处理
│   ├── features.py         # 手工特征提取（供 Baseline 使用）
│   ├── models.py           # CNN_BiLSTM, TCN, TinyMLP 网络定义
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本（输出混淆矩阵、F1）
│   ├── inference.py        # 单文件推理 / 实时推理
│   ├── augmentation.py     # 数据增强函数
│   ├── segment.py          # 动作自动切分
│   └── config.yaml         # 超参数配置文件
```

### `config.yaml` 示例

```yaml
# 数据参数
data:
  csv_dir: "data"
  window_size: 128          # 帧
  stride: 64                # 50% 重叠
  sample_rate: 87           # fps
  n_subcarriers: 64
  n_nodes: 4

# 预处理
preprocess:
  remove_zero_sc: true
  normalize: "zscore"       # minmax / zscore / none
  filter_first_word_invalid: true

# 模型
model:
  name: "cnn_bilstm"
  cnn_channels: [32, 32, 64, 128]
  lstm_hidden: 64
  lstm_layers: 2
  dropout: 0.5

# 训练
training:
  epochs: 200
  batch_size: 32
  lr: 0.001
  weight_decay: 0.0001
  early_stop_patience: 20
  class_weights: [1.0, 1.0, 1.2, 1.2, 2.5]

# 增强
augment:
  time_shift: true
  time_stretch: true
  gaussian_noise: 0.01
  amplitude_scale: 0.1
  subcarrier_mask_ratio: 0.05
```

---

## 11. 参考论文与开源实现

| 论文/项目 | 核心方法 | 可借鉴点 |
|---|---|---|
| WiFi-based human activity recognition using CNR | CNN + LSTM | 网络结构可直接复用 |
| Widar3.0 | CSI 商业级系统 | 速度估计 + 多节点融合策略 |
| ESPectre (francescopace/espectre) | MVS + NBVI + Hampel | 子载波选择、滤波器设计 |
| esp-csi (espressif/esp-csi) | esp_radar 组件 | 增益控制、ping 触发、 null data |
| DeepSeg: Deep Learning for WiFi-based Human Activity Segmentation | 端到端切分 + 分类 | 动作边界检测 |

---

## 12. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| `sit` / `stand` 混淆 | 模型区分困难 | 增加窗口长度（>2s），让 LSTM 捕捉完整动作过程；或加入方向性特征（坐下时幅度递减，起立时递增） |
| 跨 session 性能下降 | 时间/位置变化导致分布偏移 | 多 session 采集；训练时加入幅度缩放增强；使用 Domain Adaptation（如 DANN） |
| fall 样本少 | 检出率低 | 过采样 + Focal Loss + 更高的类别权重；采集时故意多采 fall |
| 四节点时间不同步 | 融合效果差 | 用 `pc_time_iso` 做最近邻对齐；或改用 Late Fusion |
| 实时推理延迟高 | 系统卡顿 | 降低窗口长度；用 TCN 替代 LSTM；或节点端提取特征，只传特征向量 |

---

> **最终建议：**
> 1. 先用 **XGBoost + 手工特征 + 四节点 Early Fusion** 在 2 天内跑通 Baseline，验证数据质量。
> 2. 确认数据可用后，主攻 **CNN_BiLSTM（方案 B）**，这是性价比最高的方案。
> 3. 如果时间充裕，对比 **TCN（方案 D）**，训练更快且效果接近。
> 4. 所有方案都要以 **Macro-F1** 和 **fall Recall** 为核心优化目标。
