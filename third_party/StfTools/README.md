# STF Tools 使用说明

## 工具列表

### 1. stf_to_json_full.py - JSON转换工具
将STF文件完全解析为JSON格式

**作用**: 解析protobuf消息，提取图像和点云数据

**用法**:
```bash
python3 stf_to_json_full.py <输入路径> [输出目录]
```

**参数**:
- `输入路径` - STF文件或run目录
- `输出目录` - 输出目录（默认: output）

**示例**:
```bash
# 解析单个文件
python3 stf_to_json_full.py lite_msg.stf output

# 解析整个run目录
python3 stf_to_json_full.py /path/to/run_dir output
```

---

### 2. stf_split_fast.py - 快速切包工具
流式切分大型STF文件，低内存不OOM

**作用**: 按时间间隔将大文件切分成小文件

**用法**:
```bash
python3 stf_split_fast.py <输入文件> [选项]
```

**参数**:
- `输入文件` - 要切分的STF文件
- `-i, --interval` - 时间间隔（秒，默认300）
- `-o, --output` - 输出目录（默认split_output）
- `-c, --chunk-size` - 读取块大小MB（默认500）

**示例**:
```bash
# 按5分钟切分
python3 stf_split_fast.py camera_jpg_img.stf -i 300 -o split_5min

# 使用更大块加速
python3 stf_split_fast.py camera_jpg_img.stf -i 300 -o split_5min -c 1000
```

---

### 3. stf_extract_images.py - 图像提取工具
从STF文件提取JPEG图像

**作用**: 提取JPEG编码的相机图像

**用法**:
```bash
python3 stf_extract_images.py <输入文件> [选项]
```

**参数**:
- `输入文件` - STF文件路径
- `-o, --output` - 输出目录（默认images）
- `-t, --threads` - 线程数（默认4）
- `-l, --limit` - 限制提取数量

**示例**:
```bash
python3 stf_extract_images.py camera_jpg_img.stf -o images -l 100
```

**注意**: 此工具适用于JPEG编码图像，H.265编码请使用下面的H.265专用工具

---

### 4. stf_extract_h265_images.py - H.265图像解码工具
从STF文件提取H.265编码图像并解码为JPEG

**作用**: H.265软解码为JPEG，支持多线程并行解码

**用法**:
```bash
python3 stf_extract_h265_images.py <输入文件> [选项]
```

**参数**:
- `输入文件` - STF文件路径
- `-o, --output` - 输出目录（默认h265_images）
- `-t, --threads` - 写入线程数（默认4）
- `-d, --decode-threads` - 解码线程数（默认4）
- `-q, --quality` - JPEG质量1-100（默认95）
- `-l, --limit` - 限制提取数量

**示例**:
```bash
# 基本用法
python3 stf_extract_h265_images.py camera.stf -o images

# 高质量+多线程
python3 stf_extract_h265_images.py camera.stf -q 98 -d 8 -t 8

# 快速预览前10张
python3 stf_extract_h265_images.py camera.stf -l 10
```

**依赖**: 需要安装 `av` 库（`pip install av`）

---

### 5. stf_extract_pointcloud.py - 点云提取工具
从STF文件提取激光雷达点云并转换为PCD格式

**作用**: 自动从 `lite_run_info.pb.bin` 读取激光雷达内外参，解析 SPIN 格式点云

**用法**:
```bash
python3 stf_extract_pointcloud.py <数据包目录> [选项]
```

**参数**:
- `数据包目录` - 包含 `lidar_data.stf` 和 `lite_run_info.pb.bin` 的目录
- `-o, --output` - 输出目录（默认pointclouds）
- `-l, --limit` - 限制提取数量
- `-b, --binary` - 使用binary格式（更小更快）
- `-v, --verbose` - 详细输出
- `--apply-extrinsics` - 应用外参变换到车辆坐标系

**示例**:
```bash
# 基本用法
python3 stf_extract_pointcloud.py /path/to/run_dir -o pointclouds

# 提取前10帧测试
python3 stf_extract_pointcloud.py /path/to/run_dir -o pointclouds -l 10 -v

# Binary格式 + 应用外参
python3 stf_extract_pointcloud.py /path/to/run_dir -o pointclouds -b --apply-extrinsics
```

---

### 6. visualize_pointcloud.py - 点云可视化工具
使用 Open3D 可视化 PCD 点云文件

**用法**:
```bash
python3 visualize_pointcloud.py <PCD文件或目录> [选项]
```

**参数**:
- `-m, --mode` - 查看模式: single/sequence/interactive
- `-c, --color` - 着色模式: height/intensity/none
- `-i, --interval` - 序列播放间隔秒数

**示例**:
```bash
# 查看单个文件
python3 visualize_pointcloud.py pointcloud.pcd

# 序列播放
python3 visualize_pointcloud.py pointclouds/ -m sequence -i 0.2
```

**依赖**: `pip install open3d numpy`

---

### 7. stf_reader.py - STF读取器
底层STF文件读取库

**作用**: 读取STF/QSTF格式文件（被其他工具调用）

---

### 8. stf_extract.py - hex提取工具
提取原始hex数据

**作用**: 调试用，提取未解析的原始数据

---

## 完整工作流程

### 场景1: 处理超大文件（推荐）

```bash
# 1. 快速切包
python3 stf_split_fast.py camera_jpg_img.stf -i 300 -o split_5min -c 1000

# 2. 提取指定时间段图像
python3 stf_extract_images.py split_5min/part_0000.stf -o images_part0

# 3. 批量处理所有切分文件
for f in split_5min/*.stf; do
    name=$(basename $f .stf)
    python3 stf_extract_images.py "$f" -o "images_$name"
done
```

### 场景2: 提取H.265编码图像

```bash
# H.265解码为JPEG（适用于新版本相机数据）
python3 stf_extract_h265_images.py camera.stf -o images -d 8

# 高质量提取
python3 stf_extract_h265_images.py camera.stf -o images -q 98 -d 8 -t 8
```

### 场景3: 提取JPEG编码图像

```bash
# 直接提取JPEG（适合旧版本或小文件）
python3 stf_extract_images.py camera_jpg_img.stf -o images -t 8
```

### 场景4: JSON转换

```bash
# 转换单个文件
python3 stf_to_json_full.py lite_msg.stf output

# 转换run目录所有STF
python3 stf_to_json_full.py /path/to/run_dir output
```

### 场景5: 点云提取与可视化

```bash
# 从数据包提取点云（自动读取内外参）
python3 stf_extract_pointcloud.py /path/to/run_dir -o pointclouds

# 提取前10帧测试
python3 stf_extract_pointcloud.py /path/to/run_dir -o pointclouds -l 10 -v

# 可视化点云
python3 visualize_pointcloud.py pointclouds/pointcloud_000000_xxx.pcd

# 序列播放
python3 visualize_pointcloud.py pointclouds/ -m sequence
```

---

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 编译proto（首次使用）
```bash
find /path/to/qcraft -name "*.proto" | \
  xargs protoc --proto_path=/path/to/qcraft --python_out=.

find onboard offboard common third_party -type d 2>/dev/null | \
  while read dir; do touch "$dir/__init__.py"; done
```

### 3. 开始使用
```bash
# 切包
python3 stf_split_fast.py input.stf -i 300 -o split_output

# 提取H.265图像
python3 stf_extract_h265_images.py input.stf -o images

# 提取JPEG图像
python3 stf_extract_images.py input.stf -o images

# 转JSON
python3 stf_to_json_full.py input.stf output
```

---

## 常见问题

**Q: 内存不足/OOM?**
- 使用 `stf_split_fast.py` 先切包
- 减小 `-c` 参数值

**Q: 处理速度慢?**
- 增大 `-c` 参数（切包）
- 增大 `-t` 参数（图像提取）
- 使用并行处理

**Q: 如何查看结果?**
```bash
# 查看图像
ls -lh images/
file images/*.jpg | head -5

# 查看JSON
ls output/
cat output/*.json | python3 -m json.tool
```

**Q: 图像无法打开？**
- H.265编码: 使用 `stf_extract_h265_images.py`
- JPEG编码: 使用 `stf_extract_images.py`
- 检查编码类型: 查看metadata中的encode_type字段

**Q: 支持哪些文件?**
- `lite_msg.stf` - protobuf消息 → JSON
- `camera_jpg_img.stf` (JPEG) - 相机图像 → JPEG
- `camera_jpg_img.stf` (H.265) - 相机图像 → H.265解码为JPEG
- `lidar_data.stf` - 激光雷达点云 → PCD

---

## 输出格式

### 切包输出
```
split_5min/
├── part_0000.stf
├── part_0001.stf
└── part_0002.stf
```

### 图像提取输出
```
images/
├── 00000000_86.jpg
├── 00000001_78.jpg
└── 00000002_83.jpg
```

### 点云提取输出
```
pointclouds/
├── pointcloud_000000_1234567890.pcd
├── pointcloud_000001_1234567891.pcd
└── pointcloud_000002_1234567892.pcd
```

### JSON转换输出
```
output/lite_msg/
├── frame_000000_qcraft_chassis.json
├── frame_000001_qcraft_imu.json
└── frame_000002_qcraft_planning.json
```

---

**更多信息**: 
- 依赖文件: `requirements.txt`
- Proto文件: `onboard/`, `offboard/`, `common/`, `third_party/`

