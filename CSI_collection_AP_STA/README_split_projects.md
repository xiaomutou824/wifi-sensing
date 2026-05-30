# AP/STA 拆分工程使用流程

当前目录下已经拆成两个独立 ESP-IDF 工程：

```text
CSI_AP/   # 烧到 ESP32-S3 A，作为 SoftAP
CSI_STA/  # 烧到 ESP32-S3 B，作为 STA + CSI 接收端
```

两个工程不再通过 `menuconfig` 切换角色，直接进入对应目录烧录。

## 1. 烧录 A 板：AP 端

```bash
cd /home/xing/project/wifi-sensing/CSI_collection_AP_STA/CSI_AP
source ~/esp/esp-idf/export.sh
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

看到下面日志说明 AP 已启动：

```text
ESP32-S3 role: AP endpoint
SoftAP started: ssid=CSI_AP_S3, channel=6
```

A 板烧完后保持供电。

## 2. 烧录 B 板：STA + CSI 接收端

```bash
cd /home/xing/project/wifi-sensing/CSI_collection_AP_STA/CSI_STA
source ~/esp/esp-idf/export.sh
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB1 flash monitor
```

看到下面日志说明 B 板已经连上 A 板并开始产生 CSI：

```text
STA connected, IP: 192.168.4.x, gateway: 192.168.4.1
CSI enabled
STA ping started
CSI_DATA,...
```

## 3. 保存 B 板数据

退出 `idf.py monitor`：

```text
Ctrl + ]
```

然后保存 STA 端串口：

```bash
cd /home/xing/project/wifi-sensing/CSI_collection_AP_STA
python3 tools/save_serial_csi.py --port /dev/ttyUSB0 --baud 921600 --label idle --node 1
```

AP 端不输出 CSI，保存数据只保存 STA 端。

默认保存 `1000` 条 CSI 行后自动停止。输出文件会写到 `data/` 目录，文件名类似：

```text
data/node1_idle_20260528_004517.csv
```

常用参数：

```bash
# 采集 walking，保存到默认 data/，1000 帧后停止
python3 tools/save_serial_csi.py --port /dev/ttyUSB0 --baud 921600 --label walking --node 1

# 改为采集 1500 帧
python3 tools/save_serial_csi.py --port /dev/ttyUSB0 --baud 921600 --label fall --node 1 --max-rows 1500

# 不自动停止，手动 Ctrl+C 结束
python3 tools/save_serial_csi.py --port /dev/ttyUSB0 --baud 921600 --label idle --node 1 --max-rows 0

# 保存到指定目录
python3 tools/save_serial_csi.py --port /dev/ttyUSB0 --baud 921600 --label idle --node 1 --out-dir data/session_01
```

## 4. 实时查看 CSI

如果想边采集边看 CSI 热图、平均幅度变化和 RSSI 曲线，用：

```bash
cd /home/xing/project/wifi-sensing/CSI_collection_AP_STA
python3 tools/live_csi_monitor.py --port /dev/ttyUSB0 --baud 921600 --label idle --node 1
```

常用参数：

```bash
# 最近 200 帧作为显示窗口，每 20 帧刷新一次
python3 tools/live_csi_monitor.py --port /dev/ttyUSB0 --label walking --node 1 --window 200 --update-every 20

# 只看图，不保存 CSV
python3 tools/live_csi_monitor.py --port /dev/ttyUSB0 --label idle --node 1 --no-save

# 只保存 CSV，不打开实时图
python3 tools/live_csi_monitor.py --port /dev/ttyUSB0 --label idle --node 1 --no-plot
```

`live_csi_monitor.py` 会保存到 `data/`，文件名包含 `_live_`。如果运行环境没有图形界面，使用 `--no-plot`。

## 5. 检查采集质量

采集完一个 CSV 后，先跑质量检查：

```bash
python3 tools/inspect_csi_settings.py data/node1_idle_20260528_004517.csv
```

重点看这些输出：

- `I/Q pairs parsed`：本项目正常应主要是 `64 pairs`
- `CSI length bytes`：正常应主要是 `128 bytes`
- `First word invalid`：`0` 占比越高越好
- `RSSI`：建议大致在 `-35 ~ -65 dBm`
- `Frame interval`：应在几十毫秒以内；如果接近 `1000 ms`，说明触发 CSI 的包太慢
- `RX sequence discontinuities`：越少越好，太多说明丢包/串口/链路可能不稳

## 6. 生成离线图

用 `visualize_csi.py` 把保存好的 CSV 画成图片：

```bash
python3 tools/visualize_csi.py data/node1_idle_20260528_004517.csv
```

默认输出到 `figures/`，会生成三张图：

- `*_amplitude_heatmap.png`：CSI 幅度热图
- `*_mean_amplitude.png`：平均幅度和帧间差分
- `*_rssi.png`：RSSI 曲线

常用参数：

```bash
# 去掉始终为 0 的子载波后再画图
python3 tools/visualize_csi.py data/node1_idle_20260528_004517.csv --drop-zero-subcarriers

# 只画前 500 帧
python3 tools/visualize_csi.py data/node1_idle_20260528_004517.csv --max-frames 500

# 只保留 64 对 I/Q 的帧，过滤异常长度
python3 tools/visualize_csi.py data/node1_idle_20260528_004517.csv --expected-iq-pairs 64

# 指定输出目录
python3 tools/visualize_csi.py data/node1_idle_20260528_004517.csv --out-dir figures/session_01
```

## 7. 推荐采集流程

每个动作建议按下面顺序执行：

```bash
# 1. 采集 1000 帧
python3 tools/save_serial_csi.py --port /dev/ttyUSB0 --baud 921600 --label idle --node 1

# 2. 检查质量
python3 tools/inspect_csi_settings.py data/node1_idle_YYYYMMDD_HHMMSS.csv

# 3. 生成图像，肉眼确认 CSI 是否稳定/有动作变化
python3 tools/visualize_csi.py data/node1_idle_YYYYMMDD_HHMMSS.csv --drop-zero-subcarriers --expected-iq-pairs 64
```

建议至少采集这些标签：

```text
idle
walking
sitting_down
standing_up
fall
```
