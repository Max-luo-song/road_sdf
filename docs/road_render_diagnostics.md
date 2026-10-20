# 路面分层渲染诊断

入口：`tools/diagnose_road_render.py`。只读取已有 checkpoint，不训练、不修改高斯参数。
默认使用本次 `sdf-heightv2/group0/checkpoint_final.pth`、五相机 2/3/5/9/12、40000 步，
检查原轨迹与左移 3 米的 0、10、40、50、79、98 帧。帧号是完整时间线的零起始索引。

## 执行

在可运行本项目训练/轨迹渲染的 CUDA 环境中，从项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/diagnose_road_render.py
```

默认结果写入项目内 `diagnostics/road_height_v2`，不写入原实验目录。
先跑单帧确认环境及对齐，再使用另一输出目录运行完整诊断：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/diagnose_road_render.py \
  --frames 50 --cam-ids 2 --traj-types left_shift_3m \
  --output-dir diagnostics/road_height_v2_smoke
```

迁移机器时可通过 `--run-dir /path/to/run` 或 `--checkpoint /path/to/group0/checkpoint_final.pth`
指定模型；checkpoint 同目录必须有保存的 `config.yaml`。数据位置用
`--opts data.data_root=/path/to/processed/training` 覆盖。不要通过 opts 改变训练相机集合、模型结构、
尺度或透明度约束，否则不再是同一渲染条件。输出目录有 manifest 时默认拒绝覆盖，可使用新目录或 `--overwrite`。
`--skip-arrays` 可以节省磁盘；`--downscale 2` 可做预览，但标线边界分析应使用默认原分辨率。

## 输出及口径

- `index.html`：所有帧的对比面板；`summary.csv`：每个视角/ROI 的统计；
  `manifest.json`：执行状态、checkpoint 信息、运行代码哈希、阈值、逐样本参考图对齐结果。
- 每个视角输出 full、road、background、no_background 四组 RGB、累计 alpha、expected depth、
  相对深度突变图。`full_composed.png` 与 `no_background_composed.png` 使用相同天空和 affine。
  所有面板不融合自车模板，统计排除自车区域；参考图验证时恢复原导出中的自车融合。
- `*_rgb_black.png` 是在黑底上的预乘颜色；`*_rgb_normalized.png` 除以累计 alpha，
  用于区分低覆盖造成的暗与颜色本身暗。alpha < 0.05 的区域用洋红标记，不强行恢复颜色。
- `visible_contributions.png` 在**完整场景原始遮挡顺序**中以 one-hot 颜色重新渲染：
  红=RoadNodes、绿=Background、蓝=其他高斯的实际可见 alpha 贡献。三个通道之和应等于完整 alpha，
  脚本自动检查最大误差 ≤ 1e-3。它是混合权重，不是最终 RGB 的各层颜色能量。
- `*_depth.png` 是相机 Z 方向的 alpha 归一化期望深度（米），各层统一 0–100m 灰度范围，
  无效区域洋红；不是沿射线距离，也不是真实表面测量。`*_expected_depth_normals.png`
  是从该深度图推导的相机坐标法线，编码为 `(n+1)/2`；边界、缺支撑及 >5% 深度突变处无效。
  它不是高斯旋转法线，也不能直接用来证明道路几何正确。
- `arrays.npz` 保留未量化 float32 RGB、alpha、深度、实际贡献、有效 mask 和期望深度法线，
  可离线重新设置阈值；`camera.json` 记录本次内外参、图像 ID、归一化时间戳。

分层使用正常 forward 收集的同一批已激活高斯，不单独重建模型或更新高度。仅移除指定类，
因此包含动态物体的完整场景和去背景场景仍保留相同动态高斯。
所有采样帧先按原脚本生成完整时间线，然后取子集，避免只用六帧重新插值改变位姿/时间。

图像坐标使用光栅化时的 pinhole 坐标；若相机原输出会恢复鱼眼畸变，本脚本不执行该重映射，
以保持 RGB/depth/alpha/ROI 一一对应，且跳过与已有鱼眼 PNG 的直接像素比较。
非鱼眼且同分辨率时，检查原实验 `novel_traj/<trajectory>_step40000/images/<frame>/<cam>_rgbs.png`，
记录 MAE、最大差和超过 2/255 的像素比例。若差异较大，应先排查代码版本/轨迹，不能直接归因。
当前工作区代码可能与训练快照不同，脚本保存实际执行代码哈希，不声称自动使用训练备份。

## 定位局部伪影

默认 `lower_half_context` 仅为下半幅上下文 ROI，**不是新视角路面真值 mask**。
路面统计基于 isolated-road alpha ≥ 0.05，并排除自车和其他类实际贡献 ≥ 0.1 的区域；
可见路面进一步要求完整场景 road 贡献 ≥ 0.05。完全缺少路面支撑的洞不会进入该统计，
因此仍需查看 alpha 图和具体 ROI，不能仅凭 CSV 排除空洞。

建议用 JSON 框选图中的毛边/黑斑，位置采用当前 pinhole 图像归一化坐标 `[x0,y0,x1,y1]`：

```json
[
  {"name": "cam9_near_patch", "camera": 9, "frame": 79,
   "traj": "left_shift_3m", "box": [0.25, 0.65, 0.7, 0.95]}
]
```

该框只是格式示例，不代表已确定的伪影位置。通过 `--roi-file /path/to/rois.json` 加入统计。
可省略 camera/frame/traj 字段，将框应用于所有匹配样本。

按如下顺序判断：

1. Road-only 的归一化 RGB 已有毛边，alpha 连续：更支持路面自身边界/纹理表达问题。
2. 黑斑对应 road alpha 降低，而归一化 RGB 正常：更支持覆盖不足，但仍要核对完整场景贡献。
3. Road-only 较干净，完整图变脏，同时背景可见贡献高、移除背景后局部改变：支持背景干扰。
4. 毛边伴随 road expected-depth 跳变：支持几何/多层混合候选原因；不等于已证明高度场错误。

暗像素默认亮度 <0.15，弱覆盖 alpha <0.9、明显背景贡献 ≥0.1、邻接相对深度跳变 >5%。
CSV 记录这些事件在暗区/可见路面中的比例和去背景前后颜色 MAE；阈值只是诊断选项，
暗像素不等于伪影（沥青/阴影也可能暗），去背景差异也不等于背景错误。
孤立层渲染移除了遮挡，不能把 road-only 和 background-only 的 RGB/深度直接相加来解释完整图。

## CPU 验证

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_road_render_diagnostics.py' -v
PYTHONPATH=. python tools/diagnose_road_render.py --help
```

单元测试验证遮挡贡献守恒、背景移除、空层、alpha 去预乘、深度有效域、平面法线、
ROI 和无有效样本时的空统计。CUDA 集成和本次实验图像需在执行环境中验证。
