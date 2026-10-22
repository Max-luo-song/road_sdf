# ExtraGS 路面实验

本分支新增独立的 `ExtraGSRoadField` 和 `configs/road_extrags.yaml`，按 [ExtraGS §3.2](https://arxiv.org/html/2508.15529v2#S3.SS2) 中明确给出的路面设计配置现有系统：共享二维 XY hash-grid 特征分别输入高度、坡度、视角颜色 MLP；SDF 为 `slope(x,y) × [z - H(x,y)]`，路面 mask 像素经 SDF CDF 体渲染做 L1 与 Eikonal 训练，Eikonal 覆盖这些射线的全部采样点。另用路面高斯在完整场景中的贡献渲染图监督语义 road mask。路面高斯的 Z 和局部法线由 SDF 无梯度更新，最终渲染只使用高斯。路面高斯只存两维切向尺度，位置、旋转、尺度和透明度固定，仅三通道 RGB 外观参数参与高斯优化。该路径不继承 e1 的初始高度网格、有界残差、LiDAR 高度锚定或覆盖率损失。

作者的[公开仓库](https://github.com/wm-research/extrags)没有训练代码，论文没有给出高斯初始化密度、透明度、法向厚度、射线采样和各损失的完整数值。本实现以已处理的 QCraft 路面点云确定 XY 范围及初始中位高度；独立的 `ExtraGSRoadNodes` 将原始路面点按 0.1 m XY 单元取均值生成高斯，固定切向尺度 0.07 m、透明度 0.95。不使用旧 `RoadNodes` 的 5 cm 交错网格、KNN 插色或标线特殊初始化。当前混合场景使用 3D gsplat，因此渲染时为两维尺度临时补 0.005 m 法向厚度；SDF 在当前高度交点附近采样并按 NeuS CDF 积分。完整图像使用 L1、SSIM、LPIPS 与 mask 监督；论文公开的 L1、SSIM、LPIPS、mask、SDF 总权重依次为 0.8、0.2、0.05、0.5、0.5，本配置按这组权重设置，禁用原深度、RoadLoss 和 Background 路面侵占正则。本次仅实现论文 RSG，不包含远场高斯、视角不确定度和生成先验。

从头训练并在 group0 完成原轨迹评估及左移 3 m 渲染：

```bash
bash scripts/qcraft/run_extrags_road.sh 20260613_102138_QCJPTDJ07384_3298_3308
```

可设置 `EXTRAGS_GPU`、`RUN_NAME_OVERRIDE`；`EXTRAGS_DRY_RUN=1` 仅检查配置和数据路径。输出使用独立的 `extrags-road` 标记。不要从 e1 checkpoint 续训，参数约束和损失不同。真实训练效果仍需在带 PyTorch、CUDA 和已处理数据的训练机上验证。
