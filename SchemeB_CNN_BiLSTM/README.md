# 方案 B：1D-CNN + BiLSTM

本目录是 CSI 人体动作识别方案 B 的代码实现，模型结构为 `1D-CNN + BiLSTM`。

模型输入统一为：

```text
[batch, time_steps, n_features]
```

其中 `time_steps` 是窗口长度，`n_features` 是子载波数、特征数或多节点拼接后的维度。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `model.py` | `CNNBiLSTM` 模型定义，支持可选 Multi-Head Attention |
| `train.py` | 训练入口，读取 `.npz` 窗口数据并保存 checkpoint |
| `predict_npz.py` | 对 `.npz` 数据做推理和评估 |
| `prepare_espfi_har.py` | 将 ESP-Fi HAR `.mat` 原始数据转换为本模型可训练的 `.npz` |
| `config.yaml` | 本项目自采 5 类动作数据的默认配置 |
| `config_espfi_har.yaml` | ESP-Fi HAR 7 类动作训练配置 |
| `requirements.txt` | 训练依赖 |

## 方式一：训练本项目自采数据

先使用项目里已有的数据预处理代码生成训练、验证、测试窗口。

```bash
cd "../data preprocessing"
python3 example_train_ready.py
```

训练脚本默认读取以下文件：

- `../data preprocessing/processed/train_windows.npz`
- `../data preprocessing/processed/val_windows.npz`
- `../data preprocessing/processed/test_windows.npz`

每个 `.npz` 文件需要包含：

- `windows`：形状为 `[N, T, C]`
- `labels`：字符串标签数组

其中 `T` 是窗口长度，`C` 是子载波/特征维度。

从项目根目录进入本目录后运行：

```bash
cd /home/xing/project/wifi-sensing/SchemeB_CNN_BiLSTM
python3 -m pip install -r requirements.txt
python3 train.py --config config.yaml
```

`config.yaml` 默认识别 5 类：

```text
idle / walking / sitting_down / standing_up / fall
```

## 方式二：训练 ESP-Fi HAR 7 类动作

ESP-Fi HAR 数据位于：

```text
../datasets/ESP-Fi-HAR/raw
```

该数据集包含 7 类动作：

| 标签 | 中文 |
| --- | --- |
| `run` | 跑 |
| `fall` | 跌倒 |
| `walk` | 走 |
| `turn` | 转身 |
| `jump` | 跳 |
| `squat` | 蹲下 |
| `arm_wave` | 挥手 |

### 1. 生成训练窗口

从项目根目录运行：

```bash
python3 SchemeB_CNN_BiLSTM/prepare_espfi_har.py \
  --raw-dir datasets/ESP-Fi-HAR/raw \
  --output-dir datasets/ESP-Fi-HAR/processed_cnn_bilstm \
  --window-size 256 \
  --stride 128 \
  --split-mode subject
```

默认设置：

- 使用 `.mat` 文件中的 `CSIamp`
- 每个原始样本为 `[950, 52]`
- 滑动窗口为 `[256, 52]`
- 步长为 `128`
- 每个样本生成 7 个窗口
- 归一化方式为 `window_zscore`
- 按参与者划分数据集，避免同一人的数据同时出现在训练集和测试集：
  - 参与者 `1-6`：训练集
  - 参与者 `7`：验证集
  - 参与者 `8`：测试集

生成结果：

```text
datasets/ESP-Fi-HAR/processed_cnn_bilstm/
├── train_windows.npz
├── val_windows.npz
├── test_windows.npz
├── summary.json
└── window_meta.json
```

当前默认转换后的规模为：

| split | 原始样本数 | 窗口数 | 单窗口形状 |
| --- | ---: | ---: | --- |
| train | 1680 | 11760 | `[256, 52]` |
| val | 280 | 1960 | `[256, 52]` |
| test | 280 | 1960 | `[256, 52]` |

### 2. 训练 7 类分类模型

在服务器或已安装 PyTorch 的环境中运行：

```bash
cd /home/xing/project/wifi-sensing/SchemeB_CNN_BiLSTM
python3 -m pip install -r requirements.txt
python3 train.py --config config_espfi_har.yaml
```

如果只想快速测试流程，可以先跑少量 epoch：

```bash
python3 train.py --config config_espfi_har.yaml --epochs 5 --batch-size 128
```

ESP-Fi HAR 的训练输出会保存到：

```text
SchemeB_CNN_BiLSTM/output_espfi_har/
```

## 训练输出

训练输出会保存到 `output/` 目录：

- `best_model.pt`
- `last_model.pt`
- `metrics.json`
- `label_map.json`
- `confusion_matrix.png`：存在验证集或测试集时生成

使用 `config_espfi_har.yaml` 时，输出目录是 `output_espfi_har/`。

## 预测与评估 NPZ 数据

预测本项目自采测试集：

```bash
python3 predict_npz.py \
  --checkpoint output/best_model.pt \
  --npz "../data preprocessing/processed/test_windows.npz"
```

预测 ESP-Fi HAR 测试集：

```bash
python3 predict_npz.py \
  --checkpoint output_espfi_har/best_model.pt \
  --npz ../datasets/ESP-Fi-HAR/processed_cnn_bilstm/test_windows.npz
```

如果要把预测结果保存为 CSV：

```bash
python3 predict_npz.py \
  --checkpoint output_espfi_har/best_model.pt \
  --npz ../datasets/ESP-Fi-HAR/processed_cnn_bilstm/test_windows.npz \
  --output-csv output_espfi_har/test_predictions.csv
```

## 输入格式

模型输入形状为 `[batch, time_steps, n_features]`，因此可以直接用于：

- 原始 64 个子载波：`[N, 128, 64]`
- 去零子载波后的数据，例如：`[N, 128, 52]`
- 四节点 early fusion 后的数据，例如：`[N, 128, 256]`
- ESP-Fi HAR 窗口数据：`[N, 256, 52]`

当前训练脚本会根据 `train_windows.npz` 自动读取 `n_features`，不需要手动修改模型输入维度。

## 其他划分方式

`prepare_espfi_har.py` 支持三种划分策略：

| 参数 | 含义 | 适用场景 |
| --- | --- | --- |
| `--split-mode subject` | 按参与者划分 | 推荐，测试跨人的泛化能力 |
| `--split-mode environment` | 按环境划分 | 测试跨场景泛化能力 |
| `--split-mode sample` | 按环境和参与者组合随机划分 | 快速实验，不推荐作为最终结果 |

跨环境示例：

```bash
python3 SchemeB_CNN_BiLSTM/prepare_espfi_har.py \
  --raw-dir datasets/ESP-Fi-HAR/raw \
  --output-dir datasets/ESP-Fi-HAR/processed_cnn_bilstm_env \
  --split-mode environment \
  --train-envs 1,2,3 \
  --val-envs 4
```

## 注意事项

- ESP-Fi HAR 官方 `.mat` 中的 `CSIamp` 已经是幅度矩阵，形状为 `[950, 52]`。
- 训练标签来自文件名第三段动作 ID，而不是 CSV 里的 `taget` 字段。
- 如果要和 ESP-Fi HAR 原文结果对比，应说明数据划分方式。随机切分通常更容易得到高准确率，但跨参与者或跨环境评估更能反映泛化能力。
- 当前模型是 CNN-BiLSTM，不是 ESP-Fi HAR 原文官方 benchmark 的 ResNet18/GRU/LSTM 等模型。官方代码已下载在 `../datasets/ESP-Fi-HAR/model_code/`，可作为对照实验。
