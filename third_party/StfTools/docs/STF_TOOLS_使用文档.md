# stf_tools 可执行程序操作指南

本文档说明如何通过 **stf_tools** 可执行程序处理 STF（QSTF）文件：转 JSON、提取图像、按时间切分等。所有功能通过子命令调用，例如：`stf_tools to-json ...`、`stf_tools extract-images ...`。

---

## 一、如何运行

- **可执行文件**（推荐）：在终端中执行 `stf_tools`（Linux/mac）或 `stf_tools.exe`（Windows），后跟子命令与参数。  
- **源码方式**：在包含 `stf_tools.py` 的目录下执行 `python3 stf_tools.py <子命令> ...`，行为与可执行文件一致。

查看支持的子命令：

```bash
stf_tools
```

查看某个子命令的用法与参数：

```bash
stf_tools <子命令> --help
```

例如：`stf_tools to-json --help`、`stf_tools extract-images --help`。

---

## 二、前置要求

1. **描述符文件 proto_descriptors.bin**  
   除 **split** 外，**to-json、extract-images、extract-h265** 均依赖与可执行文件同目录下的 `proto_descriptors.bin`。若缺失，会报错「描述符文件不存在」或「缺少 xxx 类型」。  
   **split** 不需要此文件。

2. **extract-h265 子命令**  
   若使用 **extract-h265**，运行环境需已安装：`av`、`Pillow`、`numpy`（例如 `pip install av Pillow numpy`）。  
   其他子命令无额外依赖。

---

## 三、子命令说明

### 3.1 to-json — STF 转 JSON

将 STF 中的消息解析为 JSON 文件（含相机元数据等）。

```bash
stf_tools to-json <stf文件或目录> [输出目录] [-t 线程数] [-w 进程数] [--topics TOPICS]
```

| 参数 | 含义 | 默认 |
|------|------|------|
| 输出目录 | 输出根目录 | `output` |
| `-t, --threads` | 解析线程数 | 1 |
| `-w, --workers` | 解析进程数（多进程并行） | 1 |
| `--topics` | 仅输出指定 topic，逗号分隔，如 `pose_proto,objects_proto` 或 `2,33` | 全部 |

输入为**目录**时，会对其中所有 `.stf` 文件分别转换，输出到 `输出目录/目录名/文件名/`。

示例：

```bash
stf_tools to-json lite_msg.stf
stf_tools to-json lite_msg.stf my_out -w 4
stf_tools to-json /path/to/run_dir --topics pose_proto,objects_proto
stf_tools to-json --help
```

---

### 3.2 extract-images — 提取 JPEG 图像

从 STF 中提取 **JPEG 编码**的相机图像（适用于已是 JPEG 的相机流）。

```bash
stf_tools extract-images <stf文件> [-o 输出目录] [-t 线程数] [-l 数量]
```

| 参数 | 含义 | 默认 |
|------|------|------|
| `-o, --output` | 输出目录 | `images` |
| `-t, --threads` | 线程数 | 4 |
| `-l, --limit` | 最多提取条数 | 不限制 |

示例：

```bash
stf_tools extract-images camera_jpg_img.stf -o images
stf_tools extract-images camera_jpg_img.stf -o out -l 100
stf_tools extract-images --help
```

若 STF 内为 H.265 编码，请使用 **extract-h265**。

---

### 3.3 extract-h265 — 提取并解码 H.265 图像

从 STF 中提取 **H.265 编码**的相机图像，解码为 JPEG 后保存。需安装 `av`、`Pillow`、`numpy`。

```bash
stf_tools extract-h265 <stf文件> [-o 输出目录] [-t 线程数] [-d 解码线程] [-q 质量] [-l 数量]
```

| 参数 | 含义 | 默认 |
|------|------|------|
| `-o, --output` | 输出目录 | `h265_images` |
| `-t, --threads` | 写入线程数 | 4 |
| `-d, --decode-threads` | 解码线程数 | 4 |
| `-q, --quality` | 输出 JPEG 质量 (1–100) | 95 |
| `-l, --limit` | 最多提取条数 | 不限制 |

示例：

```bash
stf_tools extract-h265 camera.stf -o images
stf_tools extract-h265 camera.stf -q 98 -d 8
stf_tools extract-h265 --help
```

---

### 3.4 split — 按时间间隔切分 STF

流式读取大 STF，按时间间隔切分为多个小 STF 文件。**不依赖 proto_descriptors.bin**。

```bash
stf_tools split <stf文件> [-i 间隔秒] [-o 输出目录] [-c 块大小MB]
```

| 参数 | 含义 | 默认 |
|------|------|------|
| `-i, --interval` | 时间间隔（秒） | 300 |
| `-o, --output` | 输出目录 | `split_output` |
| `-c, --chunk-size` | 单次读取块大小（MB） | 500 |

示例：

```bash
stf_tools split camera_jpg_img.stf -i 300 -o split_5min
stf_tools split camera_jpg_img.stf -i 300 -c 1000
stf_tools split --help
```

---

## 四、注意事项

1. **STF 格式**：工具针对 **QSTF** 格式（文件头 `QSTF`），非 QSTF 可能无法正确解析。  
2. **描述符文件**：`proto_descriptors.bin` 须与当前 proto 定义一致；仅 to-json、extract-images、extract-h265 需要，split 不需要。  
3. **磁盘空间**：to-json、extract-images、extract-h265 会生成大量文件，请确保输出目录所在磁盘空间充足。  
4. **extract-images 与 extract-h265**：JPEG 流用 extract-images，H.265 流用 extract-h265。

---

## 五、常见问题

- **报错「描述符文件不存在」或「缺少 qcraft.xxx」**  
  将正确的 `proto_descriptors.bin` 放在可执行文件（或 `stf_tools.py`）同目录下。

- **extract-h265 报错「无法导入 av」**  
  安装依赖：`pip install av Pillow numpy`。

- **未知子命令**  
  子命令名可为：`to-json` / `to_json`、`extract-images` / `extract_images`、`extract-h265` / `extract_h265`、`split`。使用 `stf_tools` 无参数可查看列表。
