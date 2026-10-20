#!/usr/bin/env python3
"""生成 STF 解析说明文档"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ─────────────────────────────────────────
# 全局样式
# ─────────────────────────────────────────
style_normal = doc.styles['Normal']
style_normal.font.name = '微软雅黑'
style_normal.font.size = Pt(10.5)
style_normal._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

def set_font(run, bold=False, size=10.5, color=None, name='微软雅黑'):
    run.font.name = name
    run.font.bold = bold
    run.font.size = Pt(size)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(doc, text, level=1):
    p = doc.add_heading(level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.runs[0] if p.runs else p.add_run(text)
    if not p.runs:
        pass
    else:
        run.text = text
    run.font.name = '微软雅黑'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    if level == 1:
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    elif level == 2:
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    elif level == 3:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return p

def add_para(doc, text, bold=False, indent=False, color=None):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(0.75)
    run = p.add_run(text)
    set_font(run, bold=bold, color=color)
    return p

def add_code(doc, text):
    """等宽代码块（灰底）"""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.right_indent = Cm(0.75)
    # 灰色底纹
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F2F2F2')
    pPr.append(shd)
    run = p.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    return p

def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 表头
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_font(run, bold=True, size=10, color=(0xFF, 0xFF, 0xFF))
        # 深蓝底色
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), '2E74B5')
        tcPr.append(shd)
    # 数据行
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg = 'FFFFFF' if ri % 2 == 0 else 'DEEAF1'
        for ci, cell_text in enumerate(row_data):
            cell = row.cells[ci]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(cell_text)
            set_font(run, size=9.5)
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), bg)
            tcPr.append(shd)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    return table

# ═══════════════════════════════════════════════
# 封面
# ═══════════════════════════════════════════════
doc.add_paragraph()
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_p.add_run('STF 文件格式解析技术说明')
run.font.name = '微软雅黑'
run.font.bold = True
run.font.size = Pt(22)
run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
run._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

doc.add_paragraph()
sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run2 = sub_p.add_run('涵盖 LevelDB SSTable 底层结构 / lite_msg / lidar_data / camera_jpg_img 三种数据文件')
run2.font.name = '微软雅黑'
run2.font.size = Pt(11)
run2.font.color.rgb = RGBColor(0x70, 0x70, 0x70)
run2._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')

doc.add_paragraph()
doc.add_page_break()

# ═══════════════════════════════════════════════
# 第一章：STF 文件概览
# ═══════════════════════════════════════════════
add_heading(doc, '一、STF 文件概览', 1)
add_para(doc, 'STF（Sensor Track File）是自动驾驶系统的传感器数据存储格式，底层采用 LevelDB SSTable 结构。一次完整的自动驾驶数据采集通常会产生三类 STF 文件：')
doc.add_paragraph()

add_table(doc,
    ['文件名', '数据类型', 'value 编码格式', '最终产物'],
    [
        ['lite_msg.stf',       '车辆系统消息（IMU/Pose/底盘/规划等）', 'Protobuf LiteMsgWrapper，tag_number 多路复用', 'JSON 消息文件'],
        ['lidar_data.stf',     '激光雷达点云',                          '自定义 SPIN 二进制格式（可能额外 Snappy 压缩）', 'PCD 点云文件'],
        ['camera_jpg_img.stf', '相机图像',                             'QIMGMETA 信封 + protobuf 元数据 + 图像裸码流',  'JPEG 图像文件'],
    ],
    col_widths=[3.8, 4.5, 6.0, 3.5]
)
doc.add_paragraph()
add_para(doc, '三种文件的 LevelDB SSTable 外层结构完全相同，差异仅在 Data Block entry 的 value 内容。')
doc.add_paragraph()

# ═══════════════════════════════════════════════
# 第二章：LevelDB SSTable 底层结构
# ═══════════════════════════════════════════════
add_heading(doc, '二、LevelDB SSTable 底层结构', 1)

add_heading(doc, '2.1 文件整体布局', 2)
add_code(doc,
'┌──────────────────────────┐\n'
'│  Data Block 0            │\n'
'├──────────────────────────┤\n'
'│  Data Block 1            │\n'
'├──────────────────────────┤\n'
'│  ...                     │\n'
'├──────────────────────────┤\n'
'│  Data Block N            │\n'
'├──────────────────────────┤\n'
'│  Meta Block (Bloom Filter)│\n'
'├──────────────────────────┤\n'
'│  MetaIndex Block         │\n'
'├──────────────────────────┤\n'
'│  Index Block             │\n'
'├──────────────────────────┤\n'
'│  Footer（固定 48 字节）   │\n'
'└──────────────────────────┘'
)
doc.add_paragraph()

add_heading(doc, '2.2 Footer（解析入口，固定 48 字节）', 2)
add_para(doc, 'Footer 位于文件末尾，是整个解析的起点：')
add_code(doc,
'偏移  长度    字段\n'
'0     变长    MetaIndex Block Handle (varint64 offset + varint64 size)\n'
'?     变长    Index Block Handle     (varint64 offset + varint64 size)\n'
'40    8字节   Magic Number = 0x57fb808b24e37b25\n'
'（不足40字节部分用0填充）'
)
add_para(doc, '解析步骤：seek 到文件末尾 -48 处 → 验证 Magic Number → 读取两个 BlockHandle。')
doc.add_paragraph()

add_heading(doc, '2.3 Block 通用读取方式', 2)
add_para(doc, '所有 Block（Index / Data / Meta / MetaIndex）读取格式相同：')
add_code(doc,
'┌───────────────────────────┐\n'
'│  Block Data（size 字节）  │  ← 压缩或未压缩的内容\n'
'├───────────────────────────┤\n'
'│  Compression Type（1字节）│  0=无压缩  1=Snappy  2=Zstd\n'
'├───────────────────────────┤\n'
'│  CRC32 校验值（4字节）     │\n'
'└───────────────────────────┘'
)
add_para(doc, '读取流程：按 Handle{offset, size} 读 size+5 字节 → 校验 CRC32 → 按 compression_type 解压 → 得到原始 Block 内容。')
doc.add_paragraph()

add_heading(doc, '2.4 Block 内部通用结构（前缀压缩 + 重启点）', 2)
add_para(doc, 'MetaIndex Block、Index Block、Data Block 三者内部二进制结构完全相同：')
add_code(doc,
'┌──────────────────────────────────────┐\n'
'│  Entry 0                             │\n'
'│  Entry 1                             │\n'
'│  ...                                 │\n'
'│  Entry N                             │\n'
'├──────────────────────────────────────┤\n'
'│  restart[0]  (uint32 LE, 4字节)      │  ← Entry 0 的字节偏移\n'
'│  restart[1]  (uint32 LE, 4字节)      │  ← 第16个Entry的偏移（完整key）\n'
'│  ...                                 │\n'
'│  num_restarts (uint32 LE, 4字节)     │\n'
'└──────────────────────────────────────┘\n'
'\n'
'每个 Entry 格式：\n'
'  shared_len    varint32   与前一个 key 共享的前缀字节数\n'
'  unshared_len  varint32   非共享部分（key_delta）长度\n'
'  value_len     varint32   value 长度\n'
'  key_delta     bytes      key 的非共享部分\n'
'  value         bytes      value 内容\n'
'\n'
'重建完整 key：full_key = prev_key[:shared_len] + key_delta'
)
add_para(doc, '重启点（Restart Point）：每隔16个 Entry 强制 shared_len=0（完整key），用于支持块内二分查找。')
doc.add_paragraph()

add_heading(doc, '2.5 三种 Block 的 KV 语义对比', 2)
add_table(doc,
    ['Block 类型', 'key 含义', 'value 含义', '谁指向它', '它指向谁'],
    [
        ['MetaIndex Block', 'Meta Block 的名字字符串\n如"filter.leveldb.BuiltinBloomFilter2"', 'Meta Block 的 BlockHandle {offset, size}', 'Footer.metaindex_handle', 'Bloom Filter Block / Stats Block'],
        ['Index Block',     '分隔键（Separator Key）\n满足：Block[i]末尾key ≤ 分隔键 < Block[i+1]首key', 'Data Block 的 BlockHandle {offset, size}', 'Footer.index_handle', '各个 Data Block'],
        ['Data Block',      'Internal Key\n= UserKey + (seq_num<<8|value_type) 8字节tag', '用户写入的真实 value 字节串\n（kTypeDeletion 时 value 为空）', 'Index Block entry.value', '无（终点）'],
    ],
    col_widths=[3.2, 5.5, 5.0, 3.5, 3.8]
)
doc.add_paragraph()

add_heading(doc, '2.6 Internal Key 结构', 2)
add_code(doc,
'Internal Key 布局（Data Block entry 的 key）：\n'
'┌───────────────────────────────┐\n'
'│  User Key（变长字节串）        │  用户原始 key\n'
'├───────────────────────────────┤\n'
'│  Sequence Number（7字节 LE）  │  全局递增写入序列号（56位）\n'
'├───────────────────────────────┤\n'
'│  Value Type（1字节）           │  1=kTypeValue  0=kTypeDeletion\n'
'└───────────────────────────────┘\n'
'后8字节合并为 uint64：(sequence_number << 8) | value_type'
)
doc.add_paragraph()

add_heading(doc, '2.7 Meta Block（Bloom Filter）', 2)
add_para(doc, 'Meta Block 不参与二分查找，作用是在读取 Data Block 之前快速排除"key 一定不存在"的情况，节省磁盘 IO：')
add_code(doc,
'每个 Data Block 对应一个 Bloom Filter bit array\n'
'filter index = Data Block 的文件 offset >> 11  （每 2KB 对应一个 filter）\n'
'\n'
'查找流程：\n'
'  1. 用 filter 探测 key → "不存在" → 直接跳过该 Data Block（零 IO）\n'
'  2. 用 filter 探测 key → "可能存在" → 读取 Data Block 精确查找\n'
'  （Bloom Filter 无假阴性，有一定假阳性率）'
)
doc.add_paragraph()

add_heading(doc, '2.8 完整查找流程', 2)
add_code(doc,
'读 Footer\n'
'  └─→ index_handle  ──→  读 Index Block\n'
'                              └─→ 二分重启点数组，找第一个 key >= target 的 entry\n'
'                                    └─→ entry.value = Data Block Handle\n'
'  └─→ metaindex_handle ──→ 读 MetaIndex Block\n'
'                                └─→ 找"filter.xxx" entry → Filter Block Handle\n'
'                                      └─→ 读 Filter Block（缓存内存）\n'
'                                            └─→ 按 offset>>11 取对应 bloom filter\n'
'                                                  ├─ 不存在 → 跳过\n'
'                                                  └─ 可能存在 → 读 Data Block\n'
'                                                                  └─→ 二分重启点\n'
'                                                                        └─→ 找 Internal Key\n'
'                                                                              ├─ kTypeValue → 返回 value\n'
'                                                                              └─ kTypeDeletion → Not Found'
)
doc.add_paragraph()

# ═══════════════════════════════════════════════
# 第三章：lite_msg.stf
# ═══════════════════════════════════════════════
add_heading(doc, '三、lite_msg.stf 解析', 1)

add_heading(doc, '3.1 Value 结构', 2)
add_para(doc, 'lite_msg 的 entry value 是 Protobuf 编码的 LiteMsgWrapper 消息，以 tag_number 实现多话题复用：')
add_code(doc,
'value 字节布局：\n'
'┌──────────────────────────────────────────────┐\n'
'│ 0x08  (1字节)  ← protobuf field1, varint类型  │\n'
'│ tag_number (varint) ← 消息类型ID              │\n'
'│ ... 其他 protobuf 字段（实际消息内容）...      │\n'
'└──────────────────────────────────────────────┘'
)
doc.add_paragraph()

add_heading(doc, '3.2 tag_number 对应的话题（部分）', 2)
add_table(doc,
    ['tag_number', '话题名称', 'tag_number', '话题名称'],
    [
        ['2',  'pose_proto',            '7',  'imu_raw_reading_proto'],
        ['8',  'pandar_packet_proto',   '9',  'obstacles_proto'],
        ['13', 'gnss_pose_proto',       '22', 'chassis_detail'],
        ['23', 'chassis',              '24', 'control_command'],
        ['49', 'localization_pose_proto','101','range_images_proto'],
    ],
    col_widths=[2.5, 5.0, 2.5, 5.0]
)
doc.add_paragraph()

add_heading(doc, '3.3 解析流程', 2)
add_code(doc,
'lite_msg.stf\n'
'  │\n'
'  ▼ LevelDB SSTable 标准解析（Footer→Index→Data Block）\n'
'  │\n'
'  ▼ entry.key 以"0"开头 → 提取 timestamp\n'
'  │\n'
'  ▼ entry.value[0] == 0x08 → 读 tag_number（varint）\n'
'  │\n'
'  ▼ 按 tag_number 查表 → 确定 protobuf 消息类型\n'
'  │\n'
'  ▼ ParseFromString(value) → 解码为具体消息\n'
'  │\n'
'  ▼ 输出 JSON 文件'
)
doc.add_paragraph()

# ═══════════════════════════════════════════════
# 第四章：lidar_data.stf
# ═══════════════════════════════════════════════
add_heading(doc, '四、lidar_data.stf（点云）解析', 1)

add_heading(doc, '4.1 Value 结构（SPIN 格式）', 2)
add_para(doc, 'lidar_data 的 entry value 是自定义二进制 SPIN 格式，value 本身可能额外套一层 Snappy 压缩：')
add_code(doc,
'SPIN 格式布局：\n'
'┌──────────────────────────────────────────┐\n'
'│ num_scans     (int32 LE, 4字节)           │ 一帧中的扫描线数量（50~5000）\n'
'│ ref_timestamp (double LE, 8字节)          │ 参考时间戳（Unix 秒）\n'
'├──────────────────────────────────────────┤\n'
'│ Scan[0] Header（从找到的有效偏移开始）    │\n'
'│   timestamp  (double, 8字节)              │ 这条扫描线的精确时间\n'
'│   azimuth    (double, 8字节)              │ 水平方位角（度）\n'
'│   pose       (56字节)                    │ 车辆位姿（暂不使用）\n'
'│   extension  (uint32, 4字节)              │ bit[31]=parity_flag\n'
'│   beam data  (从 header_offset+44 开始)  │\n'
'│     [beam 0] nr(1B) range(2B) intensity(1B) padding(1B) [多回波...]\n'
'│     [beam 1] ...\n'
'│     ...\n'
'│     [beam N-1] ...\n'
'├──────────────────────────────────────────┤\n'
'│ Scan[1] ...                              │\n'
'│ ...                                      │\n'
'└──────────────────────────────────────────┘'
)
doc.add_paragraph()

add_heading(doc, '4.2 Scan 与 Beam 的关系', 2)
add_para(doc, '激光雷达有一个旋转电机带动所有发射器水平旋转，每转到一个角度触发所有 beam 同时发射，称为一次 Scan：')
add_code(doc,
'一帧点云 = 电机转一整圈（360°）= 数百个 Scan\n'
'\n'
'Scan：时间维度  →  记录电机某一角度时刻的所有测距结果\n'
'       azimuth 不同（每条 Scan 的方位角不同）\n'
'\n'
'Beam：空间维度  →  记录某一根激光线（固定仰角）的测距\n'
'       elevation 固定（出厂内参，每颗 beam 仰角不变）\n'
'\n'
'每个 (Scan × Beam) 组合 → 一个三维点\n'
'  水平角 = Scan.azimuth + beam_az_offset（内参奇/偶偏置）\n'
'  垂直角 = elevations[beam_idx]（内参）\n'
'  距离   = range_raw × 0.004（米）'
)
doc.add_paragraph()

add_heading(doc, '4.3 坐标转换（球坐标 → 笛卡尔坐标）', 2)
add_para(doc, '坐标系：Y 轴朝前（azimuth=0°），X 轴朝右，Z 轴朝上（前-右-上 FRU 系）')
add_code(doc,
'# 方位角合成（单位：度 → 弧度）\n'
'az_rad = (scan.azimuth + az_offsets[beam_idx]) * π / 180\n'
'el_rad = elevations[beam_idx] * π / 180\n'
'r      = range_raw * 0.004   # 单位：米\n'
'\n'
'# 球坐标 → 笛卡尔\n'
'cos_el = cos(el_rad)\n'
'x = r * cos_el * sin(az_rad)   # 右方向\n'
'y = r * cos_el * cos(az_rad)   # 前方向\n'
'z = r * sin(el_rad)            # 上方向\n'
'\n'
'# 可选：外参变换到车辆坐标系（ZYX 欧拉角旋转矩阵 + 平移）\n'
'R = Rz(yaw) · Ry(pitch) · Rx(roll)\n'
'p_vehicle = R · p_lidar + [ext.x, ext.y, ext.z]'
)
doc.add_paragraph()

add_heading(doc, '4.4 解析流程', 2)
add_code(doc,
'lidar_data.stf\n'
'  │\n'
'  ▼ LevelDB SSTable 标准解析（Footer→Index→Data Block）\n'
'  │\n'
'  ▼ entry.value → 尝试 Snappy 解压（应用层额外压缩）\n'
'  │\n'
'  ▼ 校验 SPIN 格式头：50 < num_scans < 5000，时间戳合理\n'
'  │\n'
'  ▼ 滑动窗口扫描：定位每个 Scan Header（过滤间距<150字节的误判）\n'
'  │\n'
'  ▼ for each Scan:\n'
'  │    读 extension → parity_flag → 选 even/odd azimuth 偏置表\n'
'  │    从 scan_offset+44 开始读 beam data\n'
'  │    for each Beam:\n'
'  │      nr = 回波数；0<nr<10 取第一回波；nr>=10 退出\n'
'  │      range_raw(uint16) × 0.004 = r（米）\n'
'  │      球坐标 → XYZ\n'
'  │\n'
'  ▼ 输出 PCD 点云文件'
)
doc.add_paragraph()

# ═══════════════════════════════════════════════
# 第五章：camera_jpg_img.stf
# ═══════════════════════════════════════════════
add_heading(doc, '五、camera_jpg_img.stf（图像）解析', 1)

add_heading(doc, '5.1 Value 结构（QIMGMETA 信封）', 2)
add_code(doc,
'value 字节布局：\n'
'┌───────────────────────────────────────────────┐\n'
'│ "QIMGMETA"  (8字节 ASCII 魔数)                │ 标识这是一帧图像\n'
'├───────────────────────────────────────────────┤\n'
'│ meta_len    (int32 LE, 4字节)                  │ protobuf 元数据长度\n'
'├───────────────────────────────────────────────┤\n'
'│ EncodedImageMetadata (meta_len 字节 protobuf)  │\n'
'│   camera_id     → 相机 ID                     │\n'
'│   image_info.width / height → 图像尺寸         │\n'
'│   image_info.encode_type   → 0=JPEG  1=H.265  │\n'
'│   image_info.is_p_frame    → H.265 是否为P帧  │\n'
'├───────────────────────────────────────────────┤\n'
'│ image_data  (剩余所有字节)                     │ JPEG 码流 或 H.265 码流\n'
'└───────────────────────────────────────────────┘'
)
doc.add_paragraph()

add_heading(doc, '5.2 两种编码格式处理', 2)
add_table(doc,
    ['维度', 'JPEG（encode_type=0）', 'H.265（encode_type=1）'],
    [
        ['image_data 标志', r'以 \xff\xd8 开头（SOI），\xff\xd9 结尾（EOI）', 'HEVC Annex B 码流'],
        ['处理方式', '直接写入 .jpg 文件，零解码', 'PyAV 软解码 → PIL Image → JPEG 重编码'],
        ['帧依赖', '无，每帧独立', 'P帧依赖前序 I 帧（关键帧）'],
        ['性能瓶颈', '磁盘 IO', 'CPU 解码（多线程并行）'],
    ],
    col_widths=[3.0, 6.5, 6.5]
)
doc.add_paragraph()

add_heading(doc, '5.3 多线程流水线架构（H.265）', 2)
add_code(doc,
'主线程\n'
'  遍历消息，找到 H.265 帧，推入 decode_queue\n'
'       │\n'
'       ▼  decode_queue（最多50帧）\n'
'decode_worker × N 线程   ← CPU密集：PyAV 软解 H.265 → JPEG bytes\n'
'       │\n'
'       ▼  write_queue（最多100帧）\n'
'writer_worker × M 线程   ← IO密集：写 .jpg 文件到磁盘'
)
doc.add_paragraph()

add_heading(doc, '5.4 解析流程', 2)
add_code(doc,
'camera_jpg_img.stf\n'
'  │\n'
'  ▼ 检查文件头：\n'
'  │   "QSTF" → QSTF 简化格式（切包产物，见第六章）\n'
'  │   其他   → LevelDB SSTable 标准解析\n'
'  │\n'
'  ▼ 识别图像帧：entry.value[:8] == "QIMGMETA"\n'
'  │\n'
'  ▼ 解析 EncodedImageMetadata → 获取 camera_id、encode_type\n'
'  │\n'
'  ├─ encode_type=0（JPEG）\n'
'  │     image_data 直接写 .jpg\n'
'  │\n'
'  └─ encode_type=1（H.265）\n'
'        → decode_queue → PyAV 解码\n'
'        → frame.to_ndarray(rgb24) → PIL → JPEG\n'
'        → write_queue → 写 .jpg'
)
doc.add_paragraph()

# ═══════════════════════════════════════════════
# 第六章：QSTF 格式
# ═══════════════════════════════════════════════
add_heading(doc, '六、QSTF 格式（切包产物）', 1)

add_heading(doc, '6.1 来源', 2)
add_para(doc, 'QSTF 是 stf_split_fast.py 对大型图像 STF 文件做流式切包时生成的简化格式，不是原始采集格式。')
add_para(doc, '切包原理：绕开 SSTable 解析，直接扫描原始字节流中的 QIMGMETA 魔数，按时间段重新打包，输出更小的文件。')
doc.add_paragraph()

add_heading(doc, '6.2 文件结构', 2)
add_code(doc,
'┌──────────────────────────────────────┐\n'
'│ "QSTF"          (4字节魔数)           │\n'
'├──────────────────────────────────────┤\n'
'│ message_count   (uint32 LE, 4字节)   │ ← 文件关闭时回填真实数量\n'
'├──────────────────────────────────────┤\n'
'│ [帧0: QIMGMETA + meta + image_data]  │ ← 直接拷贝自原始 value\n'
'│ [帧1: QIMGMETA + meta + image_data]  │\n'
'│ ...                                  │\n'
'└──────────────────────────────────────┘\n'
'\n'
'帧间无分隔符，读取时靠扫描 "QIMGMETA" 魔数重新定位边界'
)
doc.add_paragraph()

add_heading(doc, '6.3 格式来源关系', 2)
add_code(doc,
'原始采集\n'
'  ├── lite_msg.stf          ┐\n'
'  ├── lidar_data.stf        ├── LevelDB SSTable 格式（不产生 QSTF）\n'
'  └── camera_jpg_img.stf   ┘\n'
'              │\n'
'              │ stf_split_fast.py（流式切包）\n'
'              ▼\n'
'  part_0000.stf  ┐\n'
'  part_0001.stf  ├── QSTF 简化格式\n'
'  ...            ┘'
)
doc.add_paragraph()

# ═══════════════════════════════════════════════
# 第七章：三种 STF 横向对比
# ═══════════════════════════════════════════════
add_heading(doc, '七、三种 STF 横向对比', 1)

add_table(doc,
    ['对比维度', 'lite_msg.stf', 'lidar_data.stf', 'camera_jpg_img.stf'],
    [
        ['SSTable 外层', 'LevelDB SSTable', 'LevelDB SSTable', 'LevelDB SSTable'],
        ['entry key', '"0"+timestamp UTF-8', '"0"+timestamp UTF-8', '"0"+timestamp UTF-8'],
        ['value 开头标志', '0x08（protobuf tag）', 'num_scans int32 + timestamp double', '"QIMGMETA" 8字节魔数'],
        ['value 编码', 'Protobuf LiteMsgWrapper', '自定义 SPIN 二进制', 'QIMGMETA信封+protobuf+图像裸流'],
        ['消息类型数', '几十种（tag_number 区分）', '单一（激光雷达扫描帧）', '单一（相机图像帧，可多种编码）'],
        ['Value 额外压缩', '无', '可能有额外 Snappy 压缩', '无'],
        ['最终解析结果', '各类 protobuf → JSON', '极坐标 → XYZ → PCD', 'JPEG直存 / H.265软解 → JPEG'],
        ['处理工具', 'stf_to_json_full.py', 'stf_extract_pointcloud.py', 'stf_extract_images.py\nstf_extract_h265_images.py'],
    ],
    col_widths=[3.2, 4.0, 4.5, 4.8]
)
doc.add_paragraph()

# ═══════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════
out_path = '/home/not0501/Desktop/stf_tools/STF文件格式解析说明.docx'
doc.save(out_path)
print(f'已生成：{out_path}')
