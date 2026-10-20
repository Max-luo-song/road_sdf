# 新轨迹重建快速上手（Qcraft/Chery 分支）

这份文档只覆盖你当前项目里“新轨迹”相关的最短路径：先有基础重建模型，再做新轨迹渲染（可选 DiFix3D 蒸馏修复）。

## 1. 先建立代码地图（你最需要看的文件）

- 训练入口：`tools/train.py`
- 新轨迹渲染入口：`sim_render/cam/render_novel_trajectory.py`
- 新轨迹批处理脚本（DiFix 蒸馏）：`sim_render/cam/render_novel_trajectory_online.sh`
- 新轨迹批处理脚本（简单渲染）：`sim_render/cam/render_novel_trajectory.sh`
- 轨迹生成函数集合：`utils/camera.py`
- 新视角数据集包装：`datasets/driving_dataset_novel_view.py`
- Qcraft 相机与 mix_novel_views：`datasets/qcraft/qcraft_sourceloader.py`
- 混合 novel cam 的基础行为：`datasets/base/pixel_source.py`
- 物体编辑配置（轨迹插值/新增等）：`edit_config.yaml`
- Chery 预处理说明：`docs/Chery.md`

## 2. 两条主线：先训练，再新轨迹

### 主线 A：基础场景重建训练（必须先有 checkpoint）

1. 预处理数据到 `data/.../processed/training/<scene_id>/`，结构需包含 `images/`, `lidar_pose/`, `extrinsics/`, `intrinsics/` 等。
2. 运行训练（入口是 `tools/train.py`），得到 `checkpoint_final.pth`。
3. 训练配置里如果启用 `mix_novel_views`（例如 `configs/datasets/qcraft/11cams_novel_lidar.yaml`），会把 `mix_novel_cams` 作为额外相机参与训练，`num_cams` 会变成 `len(cameras)+len(mix_novel_cams)`。

### 主线 B：新轨迹渲染（可选 DiFix 蒸馏）

1. 用训练好的 `checkpoint_final.pth` 跑 `sim_render/cam/render_novel_trajectory.py`（推荐批处理入口：`sim_render/cam/render_novel_trajectory_online.sh`）。
2. `--traj_types` 决定轨迹类型，轨迹实现都在 `utils/camera.py`（如 `left_shift_1m`, `change_lane_2m`, `original_traj`）。
3. 输出目录在：
   - 普通渲染：`<log_dir>/novel_traj/<traj_type>_step<step>[_suffix]/`
   - 开启 DiFix 蒸馏：还会生成 `<log_dir>/difix_distill_all_frames_<tag>/`（子采样 debug PNG）及最终 checkpoint

**蒸馏方案详解**（Progressive 混合 stage、帧子采样、评测全帧等）见 **[NovelTrajectoryDistillation.md](./NovelTrajectoryDistillation.md)**。

## 3. 最小可运行命令

不带 DiFix（先验证链路）：

```bash
export PYTHONPATH=$(pwd)
python sim_render/cam/render_novel_trajectory.py \
  --resume_from <你的checkpoint_final.pth> \
  --traj_types original_traj left_shift_1m \
  --cam_ids 0 1 2 3 5 6 7 9 10 11 12 \
  --downscales 1 1 1 1 1 1 1 1 1 1 1 \
  --fps 10 \
  --render_rgb \
  --save_images
```

带 DiFix 蒸馏（当前推荐入口）：

```bash
bash sim_render/cam/render_novel_trajectory_online.sh
```

参数与方案说明见 [NovelTrajectoryDistillation.md](./NovelTrajectoryDistillation.md)。

## 4. 你这个分支的关键行为（和上游常见差异）

- `render_novel_trajectory.py` 已集成 DiFix3D 渐进式混合蒸馏：新轨迹 stage 做 DiFix 修复 + L1 蒸馏，原轨迹 stage 做 GT replay；详见 [NovelTrajectoryDistillation.md](./NovelTrajectoryDistillation.md)。
- 蒸馏训练默认 `distill_frame_stride=3` 子采样帧以节省算力；stage 后视频渲染与正式评测使用全帧。
- Legacy 模式下蒸馏步数会被自动抬高到至少 `样本数 × 100`；Progressive 模式（当前默认）由 `distill_stage_repeats` 决定步数。
- `--difix_use_original_traj_ref` + `--difix_ref_video_dir` 可用原轨迹 GT 视频作为 DiFix ref image。

## 5. 常见坑（建议先检查）

- `--cam_ids` 与 `--downscales` 长度必须一致，否则直接断言失败。
- `render_novel_trajectory.sh` 里 `ckpt_path` 指向了 `scene_reconstruction-main` 路径，与你当前仓库目录 `scene_reconstruction-ga` 不一致时会找不到模型。
- 开启 `--enable_difix_distill` 时，`--traj_types` 不能为空，否则脚本会报错退出。
- 设定了 `difix_use_original_traj_ref=true` 时，需配置 `difix_ref_video_dir`（或 `difix_ref_video_path`）。

## 6. 建议的阅读顺序（30 分钟版）

1. `sim_render/cam/render_novel_trajectory_online.sh`：先看实际传了哪些参数。  
2. `sim_render/cam/render_novel_trajectory.py`：看 `main()` 里 distill 分支和 `render_trajectory()`。  
3. [NovelTrajectoryDistillation.md](./NovelTrajectoryDistillation.md)：理解完整蒸馏方案与评测流程。  
4. `utils/camera.py`：看 `get_interp_novel_trajectories()` 支持哪些 `traj_type`。  
5. `datasets/driving_dataset_novel_view.py`：看轨迹如何转成每帧渲染输入。  
6. `datasets/qcraft/qcraft_sourceloader.py`：看相机加载和 `mix_novel_views` 行为。

