# e1: SDF 几何 + 路面高斯外观

基于 `codex/e0`：路面高斯的高度继续同步到可学习 SDF 高度，局部 Z 轴继续对齐 SDF 法线。辅助 SDF 射线渲染仍给几何网络提供图像、LiDAR 和 Eikonal 监督。

`configs/road_e1.yaml` 放开路面高斯的三个尺度和不透明度：不再限制 XY/Z 尺度、不再将路面透明度夹到 0.2，也不再屏蔽尺度 Z 梯度。增密后新高斯保留其尺度；高度和朝向仍由 SDF 同步。

参考 `road_v1_plus`，训练时单独渲染 RoadNodes，使用有效路面 mask 监督其去预乘颜色、覆盖率和它在完整场景中的贡献占比。颜色损失使用当前仿射变换，但不更新仿射参数。标线像素有额外权重。对应损失为 `road_color_loss`、`road_coverage_loss`、`road_contribution_loss`；`road_field_*` 损失继续存在。

运行 group0 训练、评估以及原轨迹和左移 3 米渲染：

```bash
bash scripts/qcraft/run_e1.sh 20260613_102138_QCJPTDJ07384_3298_3308
```

默认 pipeline 的 `scripts/qcraft/common.sh` 也指向 `configs/road_e1.yaml`。可用 `RUN_NAME_OVERRIDE`、`E1_GPU`、`E1_DRY_RUN=1` 调整运行。请从头训练；旧实验 checkpoint 的尺度与透明度参数语义及优化过程不同。
