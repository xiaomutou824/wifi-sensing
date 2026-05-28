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
