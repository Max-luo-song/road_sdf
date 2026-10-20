# 路面高度学习对照实验

目标：在现有路面高斯的外观表达能力相同的条件下，比较固定高度与可学习高度对
原轨迹及横移轨迹的影响。原配置不启用新分支。

参考：[ExtraGS §3.2](https://arxiv.org/html/2508.15529v2#S3.SS2)。
这是根据论文设计的适配实现，不声称复现作者未发布的代码。

## 实现与梯度路径

`models/road_surface.py::RoadSurfaceField` 使用纯 PyTorch 二维多分辨率 hash-grid
和小型 MLP，避免增加 tiny-cuda-nn 依赖：

```text
初始 RoadNodes 点 → 有效高度网格 H₀(x,y)
XY → 几何 hash-grid → MLP → ΔH、坡度因子 a
H = H₀ + 0.5m × tanh(ΔH)
d(x,y,z) = a(x,y) × [z - H(x,y)]

路面像素 → 相机射线 → 局部带内 SDF CDF 体渲染 → RGB 重建/覆盖率损失
                     ↑                         ↑
                 几何网络                独立颜色网络

原始路面 LiDAR → 高度 Huber 锚定损失
SDF 有限差分梯度 → Eikonal 损失

RoadNodes.Z ← stop_gradient(H(RoadNodes.XY))
现有高斯渲染 → 现有图像损失 → 高斯颜色/SH、透明度、切向尺度
Background → 同源高度查询 → detached 侵占置信度 → 原透明度正则/增密门控
```

几何和辅助颜色使用不同的 hash-grid，颜色网络的特征梯度不能直接修改几何特征。
图像损失仍能经体渲染权重更新几何；这不是高度与颜色完全隔离，也不能保证消除
颜色吸收几何误差的歧义。相机参数在辅助分支中 detach，防止该分支改变相机位姿。

初始网格按有效邻居做归一化双线性查询，无有效支撑处不应用残差或 Background
抑制；这会限制可修正区域，不能靠本实现恢复完全未观测的侧方路面。网格适用于
单层路面，不适用于同一 XY 上下重叠的道路。

辅助渲染在初始高度面的估计交点附近沿射线采样，使用固定的上下高度带；边界
不依赖当前可学习高度，因此两组的初始射线采样相同。过滤向上、接近水平、超出
距离范围、无路面支撑及动态/自车/天空像素。采样带必须大于高度残差上限。
这里是局部带内的 NeuS 风格 CDF 合成，不是完整的自适应 NeuS 采样器。

## 两个配置

| 配置 | 几何 | 辅助颜色分支 | 最终高斯外观 |
| --- | --- | --- | --- |
| `configs/road_height_fixed.yaml` | 高度残差和坡度网络冻结 | 训练 | 与 learned 相同 |
| `configs/road_height_learned.yaml` | 图像 + LiDAR + Eikonal 训练 | 训练 | 与 fixed 相同 |

两个配置通过 `_base_` 继承现有相机/LiDAR 配置，`tools/train.py` 在数据配置和
CLI 覆盖之前解析继承，保存到输出目录的 `config.yaml` 已展开，可继续用现有评估入口。

默认时序：

- 0–999 步：原 RoadNodes 高度不变，辅助场预训练。
- 从 1000 步开始：两组 RoadNodes 都查询同一构建方式的初始网格，learned 额外加
  可学习残差；Background 同步查询该高度场。
- 从 30000 步开始：辅助分支停止更新，剩余步数训练高斯外观。
- 每次高斯渲染前及 optimizer 更新后，无梯度同步 Z；保存的高斯已经带有最新高度。
  评估/导出不查询高度网络，直接渲染保存的高斯。

注意：网格插值不保证与原高斯初始 Z 逐点完全相等，因此 fixed 才是 learned 的
主要匹配对照；原 `omnire_extended_cam_lidar.yaml` 可作为第三组评估插值/接口改变的影响。
两组的网络初始化、辅助采样使用独立随机流，避免改变原高斯初始化和图像采样随机流。
数据、相机分组、seed、训练步数、增密设置和评估轨迹也必须相同。两组训练中仍可能
因图像梯度变化出现不同的 Background 增密，这是高度变化的下游影响。

## 运行

沿用现有训练命令的数据、相机分组及路径参数，仅替换 `--config_file` 并区分 run name：

```bash
# 在项目根目录运行；末尾追加现有的 dataset/data/相机分组等参数。
python tools/train.py --config_file configs/road_height_fixed.yaml --run_name road-height-fixed
python tools/train.py --config_file configs/road_height_learned.yaml --run_name road-height-learned
```

默认继承的数据集仍是原配置里的 `waymo/3cams`；使用 QCraft 时，需传入现有的
`dataset=qcraft/11cams_lidar` 及场景路径等参数，不能直接省略。
可通过 CLI 调整 `model.RoadNodes.networks.road_surface.*`。
`scripts/qcraft/common.sh` 已将默认配置切为 `configs/road_height_learned.yaml`，
因此原 pipeline 命令直接运行高度学习版本，启动日志会显示训练配置路径：

```bash
bash scripts/qcraft/pipeline.sh 20260613_102138_QCJPTDJ07384_3298_3308 3298 3308
```

训练前后使用匹配的实验配置。新增实验 checkpoint 包含网络和动态大小的高度网格、
LiDAR 缓冲区；恢复使用输出目录中保存的配置。旧基线 checkpoint 不含这些状态，
不能严格加载为启用高度场的模型，本轮实验应从相同数据重新初始化。
训练器仍沿用项目现有的 model-only 恢复语义，不恢复 Adam 状态。

## 观察什么

- 损失：`road_field_rgb`、`road_field_coverage`、`road_field_lidar`、
  `road_field_eikonal`；固定组没有可训练几何正则项。
- 诊断：`road_field_ray_fraction` 应明显大于零，否则检查坐标轴、路面 mask、
  支持范围和 `far`；`road_field_opacity` 应逐渐接近一。
- 高度：`road_field_height_delta_abs`；同时检查高度误差而非仅观察高度是否变化。
- `road_field_holdout_mae` 和 `road_field_initial_holdout_mae` 使用未参与新增 LiDAR
  锚定损失的点，但这些点参与过原路面初始化，所以**不是独立几何测试集**，不能
  据此声称几何泛化改善。需要独立扫描或预先排除的帧/区域进行严格验证。
- 原轨迹比较整体/路面质量；相同横移轨迹比较标线重影、连续性、路面波纹及路缘
  误抑制。没有横移真值时只作定性对比，不能用训练 RGB 下降证明几何更准确。
- 记录单步耗时和峰值显存，辅助射线体渲染会增加训练成本。

## 验证

v2 修复了 `MultiTrainer` 文件末尾重复定义 `compute_losses/compute_metrics` 覆盖
SDF 接入的问题。pipeline 的输出标记改为 `sdf-heightv2`；从头运行原命令，不要
从 v1 的 40000 步最终 checkpoint 续训（已经超过辅助场冻结步数）。首次有效
训练会打印 `Road surface training active` 及接入的损失名称。

```bash
python -m unittest discover -s tests -p 'test_road_surface*.py' -v
PYTHONPATH=. python tests/test_road_surface_prior.py
```

CPU 测试覆盖距离/深度一致性、图像梯度到几何和颜色、已知纹理下的合成高度恢复、
固定高度对照、支持域、侵占梯度隔离、Eikonal 反传、状态恢复、mask/冻结及配置继承。
新增训练器回归测试编译原始类定义，验证实际方法解析、损失汇总、反传、optimizer
更新、RoadNodes 高度同步和指标接入；仅替换需要 CUDA 光栅化的父类损失。
真实多视角的收敛、CUDA 光栅化集成和性能仍需在训练环境中验证。
