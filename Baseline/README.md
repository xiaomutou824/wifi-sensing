# CSI 动作识别 Baseline（传统机器学习）

基于 XGBoost / Random Forest 的快速 Baseline，用于验证 CSI 数据是否包含可区分的动作信息。

## 依赖安装

```bash
cd Baseline
pip install -r requirements.txt
```

## 快速开始

### 1. 确认有采集好的数据

确保 `../CSI collection/data/` 目录下有多个 CSV 文件，且标签名是动作名（如 `idle`、`walking`、`sitting_down` 等）：

```
node1_idle_20260526_122147.csv
node1_walking_20260526_123015.csv
node1_sitting_down_20260526_123530.csv
...
```

### 2. 修改配置（可选）

编辑 `config.yaml`：
- 如果数据路径不对，修改 `data.csv_paths`
- 如果要做 **Leave-One-Session-Out (LOSO)** 交叉验证，设 `training.loso_cv: true`
- 如果数据量少（只有 1-2 天的 session），设 `training.loso_cv: false`

### 3. 训练

```bash
python train_baseline.py
```

输出示例：
```
Found 15 CSV files
Train: 10 files, Val: 3 files, Test: 2 files

[1/3] Processing training set...
[Preprocess] Loading ...
  Valid frames: 3250, subcarriers: 52

Feature shape: (48, 37)

==================================================
Training: xgboost
Class weights: {0: 1.0, 1: 1.0, 2: 1.2, 3: 1.2, 4: 3.0}

[Validation] Results:
  Accuracy:      0.8125
  Macro F1:      0.7854
  Macro Prec:    0.7981
  Macro Recall:  0.7912

Classification Report:
              precision    recall  f1-score   support

        idle     0.9167    0.8462    0.8800        13
     walking     0.8000    0.8889    0.8421         9
 sitting_down  0.6667    0.8000    0.7273         5
 standing_up   0.7500    0.6000    0.6667         5
        fall   1.0000    0.5000    0.6667         2

[Test] Results:
  Accuracy:      0.8750
  Macro F1:      0.8235
...

Saved model to output/model_xgboost.pkl
```

### 4. 查看结果

训练完成后，`output/` 目录下会生成：

| 文件 | 说明 |
|---|---|
| `model_xgboost.pkl` | 训练好的 XGBoost 模型 |
| `model_random_forest.pkl` | 训练好的 Random Forest 模型 |
| `cm_xgboost.png` | XGBoost 测试集混淆矩阵 |
| `cm_random_forest.png` | RF 测试集混淆矩阵 |
| `feature_importance_xgboost.png` | XGBoost 特征重要性图 |

### 5. 对新文件做预测

```bash
python example_predict.py \
  "../CSI collection/data/node1_walking_20260526_123015.csv" \
  output/model_xgboost.pkl
```

输出：
```
Prediction Result:
  File: ../CSI collection/data/node1_walking_20260526_123015.csv
  Total windows: 16
  Majority vote: walking (confidence: 87.5%)

Distribution:
  walking: 14 (87.5%)
  idle: 2 (12.5%)
```

---

## 数据质量判断标准

| 指标 | 数据可用 | 需要优化 |
|---|---|---|
| **Test Accuracy** | > 70% | < 60% |
| **Test Macro-F1** | > 65% | < 55% |
| **fall Recall** | > 80% | < 50% |

- 如果 **Accuracy > 70%**：数据有动作信息，可以继续上深度学习模型
- 如果 **Accuracy < 60%**：先不要写复杂模型，先检查采集流程（距离、角度、ping 源唯一性）
- 如果 **fall 经常被误判为 walking**：说明 fall 的"短时剧烈变化"特征不够明显，可能需要更短的窗口或更多 fall 样本

---

## 配置速查

### 窗口大小 (`window.size`)

| 值 | 时长 (@87fps) | 适用场景 |
|---|---|---|
| 64 | ~0.7s | fall（极短动作） |
| 128 | ~1.5s | 推荐，通用 |
| 256 | ~3.0s | walking（长动作） |

### 交叉验证方式 (`training.loso_cv`)

| 场景 | 建议 |
|---|---|
| 有 **3 天以上** 的数据 | `true`：做 LOSO，验证跨时间泛化 |
| 只有 **1-2 天** 的数据 | `false`：按文件划分 train/val/test |

### 模型选择 (`models.*.enabled`)

| 模型 | 特点 | 建议 |
|---|---|---|
| XGBoost | 通常效果最好 | **必开** |
| Random Forest | 不易过拟合，可解释 | 对比用 |
| LightGBM | 训练快，需安装 | 可选 |

---

## 与深度学习的关系

Baseline 的目标不是达到最高准确率，而是**快速验证数据可用性**：

```
Baseline Accuracy > 70%  ──►  数据有信号，可以训练 CNN_BiLSTM
Baseline Accuracy < 60%  ──►  先调硬件/采集流程，不要急着上深度学习
```

通常 XGBoost Baseline 能达到 75-85%，而 CNN_BiLSTM 能在此基础上再提升 5-15%。
