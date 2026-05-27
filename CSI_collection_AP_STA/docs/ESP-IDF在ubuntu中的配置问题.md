### 在当前终端窗口中，激活 ESP-IDF 的开发环境

```
source /home/xing/.espressif/v5.5.4/esp-idf/export.sh
```

把 idf.py 的路径加入到系统 PATH（这样你才能找到它）。

激活 ESP-IDF 专用的 Python 虚拟环境（确保依赖库不冲突）。

设置好 ESP32 芯片的交叉编译工具链路径。

每次打开新窗口都要敲这个长长的命令很麻烦。你可以把它加到你的终端配置文件里，一劳永逸：

```
echo "source /home/xing/.espressif/v5.5.4/esp-idf/export.sh" >> ~/.bashrc
```

这样以后每次打开终端，ESP-IDF 环境都会自动激活，省时省力！

### Braille TTY驱动抢占串口问题

```
sudo apt remove brltty
```

brltty 在系统启动时会自动加载，并会尝试“扫描”所有的 USB 串口设备（如 /dev/ttyUSB0）。它会误以为你的 USB 转串口（CH341）是一个盲文显示器，于是强行抢占了这个设备。

后果就是： 你的 ESP32 开发板通过 USB 连接后，Linux 无法正常生成 /dev/ttyUSB0，或者即便生成了，你的 idf.py flash 也无法访问它（会报 Permission denied 或者设备忙）。这就是为什么之前推荐你删除它的原因。

### 手动挂载，实现主机和虚拟机中ubuntu实现共享文件夹

如果你的 /mnt/ 下面没有 hgfs 文件夹，或者里面是空的，说明 Ubuntu 没有成功自动挂载。请在终端（Ctrl+Alt+T）里直接执行下面这条命令：

```
sudo vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
```

执行完这条命令后，再次回到刚才的路径（/mnt/hgfs），就能看到你的文件夹了。
