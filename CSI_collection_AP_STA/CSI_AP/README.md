# CSI_AP

ESP32-S3 A 板工程：只负责创建固定 SoftAP。

默认参数：

```text
SSID: CSI_AP_S3
Password: 12345678
Channel: 6
Gateway: 192.168.4.1
```

烧录：

```bash
cd CSI_AP
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

看到 `SoftAP started` 后保持 A 板供电，再烧录/启动 `CSI_STA`。
