# CSI_STA

ESP32-S3 B 板工程：连接 `CSI_AP`，开启 CSI，并 ping AP 网关生成稳定下行 CSI 包。

默认参数必须和 AP 一致：

```text
SSID: CSI_AP_S3
Password: 12345678
Channel: 6
Ping rate: 30 Hz
```

烧录：

```bash
cd CSI_STA
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB1 flash monitor
```

看到 `CSI_DATA,...` 就说明正在输出 CSI。

保存数据时退出 monitor，然后在上一级工程使用保存脚本：

```bash
python3 tools/save_serial_csi.py --port /dev/ttyUSB1 --baud 921600 --label idle --node 1
```
