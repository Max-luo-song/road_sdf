<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/media/logo.png">
    <img alt="Logo" src="docs/media/logo_clipped.png" width="700">
  </picture>
</p>
<p align="center">
A 3DGS framework for omni urban scene reconstruction and simulation!
</p>

<p align="center">
    <!-- project -->
    <a href="https://ziyc.github.io/omnire/"><img src="https://img.shields.io/badge/Project-Page-FFFACD" height="28"/></a>
    <!-- paper -->
    <a href="https://arxiv.org/abs/2408.16760">
        <img src='https://img.shields.io/badge/arXiv-Paper-E6E6FA' height="28"/>
    </a>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/08e6c613-f61a-4d0d-a2a9-1538fcd4f5ff" width="49%" style="max-width: 100%; height: auto;" />
  <img src="https://github.com/user-attachments/assets/d2a47e7d-2934-46de-94d6-85ea8a52aba6" width="49%" style="max-width: 100%; height: auto;" />
</p>

## About
DriveStudio is a 3DGS codebase for urban scene reconstruction/simulation. It offers a system with multiple Gaussian representations to jointly reconstruct backgrounds, vehicles, and non-rigid categories (pedestrians, cyclists, etc.) from driving logs. DriveStudio also provides a unified data system supporting various popular driving datasets, including [Waymo](https://waymo.com/open/), [PandaSet](https://pandaset.org/), [Argoverse2](https://www.argoverse.org/av2.html), [KITTI](http://www.cvlibs.net/datasets/kitti/), [NuScenes](https://www.nuscenes.org/), and [NuPlan](https://www.nuscenes.org/nuplan).

This codebase also contains the **official implementation** of:
  > **OmniRe: Omni Urban Scene Reconstruction** <br> [Project Page](https://ziyc.github.io/omnire/) | [Paper](https://arxiv.org/abs/2408.16760) <br> [Ziyu Chen](https://ziyc.github.io/), [Jiawei Yang](https://jiawei-yang.github.io/), [Jiahui Huang](https://huangjh-pub.github.io/), [Riccardo de Lutio](https://riccardodelutio.github.io/), [Janick Martinez Esturo](https://www.jme.pub/), [Boris Ivanovic](https://www.borisivanovic.com/), [Or Litany](https://orlitany.github.io/), [Zan Gojcic](https://zgojcic.github.io/), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Marco Pavone](https://stanford.edu/~pavone/), [Li Song](https://medialab.sjtu.edu.cn/author/li-song/), [Yue Wang](https://yuewang.xyz/)

# 🎉 Try your own projects/research on DriveStudio!
### 🔥 Highlighted implementations

Our codebase supports two types of Gaussian trainers:

1. Single-Representation trainer (single Gaussian representation for the entire scene):
   - Deformable Gaussians
   - Periodic Vibration Gaussians

2. Multi-Representation trainer (Gaussian scene graphs trainer):
   - Background: Static Gaussians (Vanilla Gaussians)
   - Vehicles: Static Gaussians
   - Humans: SMPL-Gaussians, Deformable Gaussians
   - Other non-rigid categories: Deformable Gaussians

**Implemented methods:**

| Method Name | Implementation | Trainer Type | Gaussian Representations |
|-------------|----------------|--------------|--------------------------|
| [OmniRe](https://ziyc.github.io/omnire/) | Official | Multi | • Static Gaussians: Background, Vehicles<br>• SMPL Gaussians: Pedestrians (majority)<br>• Deformable Gaussians: Cyclists, far-range pedestrians, other non-rigid categories |
| [Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians) | Unofficial | Single | • Deformable Gaussians: Entire scene |
| [PVG](https://github.com/fudan-zvg/PVG) | Unofficial | Single | • Periodic Vibration Gaussians: Entire scene |
| [Street Gaussians](https://github.com/zju3dv/street_gaussians) | Unofficial | Multi | • Static Gaussians: Background, Vehicles |

We extend our gratitude to the authors for their remarkable contributions. If you find these works useful, please consider citing them.

### 🚗 Dataset Support
This codebase provides support for popular driving datasets. We offer instructions and scripts on how to download and process these datasets:

| Dataset | Instruction | Cameras | Sync Frequency | Object Annotation |
|---------|-------------|---------|----------------|-------------------|
| 奇瑞 | [数据预处理指南](docs/Chery.md) | 7 cameras | 10Hz | ✅ |
| Waymo | [Data Process Instruction](docs/Waymo.md) | 5 cameras | 10Hz | ✅ |
| NuScenes | [Data Process Instruction](docs/NuScenes.md) | 6 cameras | 2Hz (up to 10Hz*) | ✅ |
| NuPlan | [Data Process Instruction](docs/Nuplan.md) | 8 cameras | 10Hz | ✅ |
| ArgoVerse | [Data Process Instruction](docs/ArgoVerse.md) | 7 cameras | 10Hz | ✅ |
| PandaSet | [Data Process Instruction](docs/Pandaset.md) | 6 cameras | 10Hz | ✅ |
| KITTI | [Data Process Instruction](docs/KITTI.md) | 2 cameras | 10Hz | ✅ |

*NOTE: For NuScenes data, LiDAR operates at 20Hz and cameras at 12Hz, but keyframes (with object annotations) are only at 2Hz. We provide a method to interpolate annotations up to 10Hz.

### ✨ Functionality

<details>
<summary>Click to expand functionality details</summary>

We have implemented interesting and useful functionalities:

1. **Flexible multi-camera training:** Choose any combination of cameras for training - single, multiple, or all. You can set these up by **SIMPLY** configuring your selection in the config file.

2. **Powered by gsplat** Integrated [gsplat](https://github.com/nerfstudio-project/gsplat) rasterization kernel with its advanced functions, e.g. absolute gradients, anti-aliasing, etc.

3. **Camera Pose Refinement:** Recognizing that camera poses may not always be sufficiently accurate, we provide a method to refine and optimize these poses.

4. **Objects' GT Bounding Box Refinement:** To address noise in ground truth boxes, we've added this feature to further improve accuracy and robustness.

5. **Affine Transformation:** This feature handles camera exposure and other related issues, enhancing the quality of scene reconstruction. 

6. ...

These functionalities are designed to enhance the overall performance and flexibility of our system, allowing for more accurate and adaptable scene reconstruction across various datasets and conditions.
</details>

## 📢 Updates

**[Aug 2024]**  Release code of DriveStudio.

## 🔨 Installation

Run the following commands to set up the environment:

```shell
# Clone the repository with submodules
git clone --recursive https://github.com/ziyc/drivestudio.git
cd drivestudio

# Create the environment
conda create -n drivestudio python=3.9 -y
conda activate drivestudio
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast

# Set up for SMPL Gaussians
cd third_party/smplx/
pip install -e .
cd ../..
```

## 📊 Prepare Data
We support Chery and most popular public driving datasets. Detailed instructions for downloading and processing each dataset are available in the following documents:

- 奇瑞: [数据预处理指南](docs/Chery.md)
- Waymo: [Data Process Instruction](docs/Waymo.md)
- NuScenes: [Data Process Instruction](docs/NuScenes.md)
- NuPlan: [Data Process Instruction](docs/Nuplan.md)
- ArgoVerse: [Data Process Instruction](docs/ArgoVerse.md)
- PandaSet: [Data Process Instruction](docs/Pandaset.md)
- KITTI: [Data Process Instruction](docs/KITTI.md)

## 🚀 Running
### Training
```shell
export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/3cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

- To run other methods, change `--config_file`. See `configs/` for more options.
- Specify dataset and number of cameras by setting `dataset`. Examples: `waymo/1cams`, `waymo/5cams`, `pandaset/6cams`, `argoverse/7cams`, etc.
  You can set up arbitrary camera combinations for each dataset. See `configs/datasets/` for custom configuration details.
- For over 3 cameras or 450+ images, we recommend using `omnire_extended_cam.yaml`. It works better in practice.
### Evaluation
```shell
python tools/eval.py --resume_from $ckpt_path
```

# Background Editing & Particle Construction Pipeline

Weather Edit的天气编辑逻辑为：
1. 先编辑原始图片天气，得到新天气图片（Background Editing）
2. 使用新天气图片替换原始图片，并结合场景其他数据重建3DGS（OmniRe）
3. 再对重建好的3DGS进行渲染，在渲染的过程中添加动态粒子效果（Particle Construction）

## 🚀 Background Editing

Edit backgrounds with **snowy**, **rainy**, and **foggy** effects.

------------------------------------------------------------------------

### **1. Download Pretrained Model**

-   **Pretrained Model**\
    👉 [Download](https://drive.google.com/file/d/1r38vaV7lb4tFVyq6n3twoDwa5qhqOlti/view?usp=sharing) and place it in the `WeatherEdit/background_editing/ckpts/` folder.

    👉 [Download](https://huggingface.co/stabilityai/sd-turbo) and place it in the `WeatherEdit/background_editing/ckpts/` folder.

------------------------------------------------------------------------

### **2. Run Inference**
Our default setup is for multi-view datasets.
But if you're working with just single-view and single-frame images, we've got you covered!
Please prepare your images and mask following the sample dataset structure with filename suffix `_0`, then use `--dataset custom` for inference.

`--weather_type` only support "snowy", "rainy", "foggy"

当你使用qcraft数据时，你需要额外输入两个参数 `--data_root` 和 `--cams`，分别对应数据文件夹所在位置和你需要编辑的视角，我们建议您选择前方摄像头：[2, 5, 9]；后方摄像头：[6, 10, 12]。且建议您把输出路径设置为data-root所在路径，方便后续场景重建。

#### **Multi-View & Multi-Frame**

``` bash
cd scene_reconstruction/WeatherEdit/background_editing

# qcraft dataset
python src/inference.py     --output_dir "../../../data/qcraft/processed/training/20251025_163358_QCOYSD504206_1595_1610/images_snowy"     --dataset qcraft     --weather_type snowy  --data_root "../../../data/qcraft/processed/training/20251025_163358_QCOYSD504206_1595_1610/images" --cams 2 5 9
```

生成后的图片会保存在`output_dir/images_{weather_type}`。

------------------------------------------------------------------------

## ❄️ Particle Construction

Driving scene reconstruction and generate **dynamic 3D particles** (e.g., snow, rain, fog) in those scenes.

在对图片进行天气编辑之后，我们需要使用编辑后的图片做场景重建。

------------------------------------------------------------------------

### **1. 3D Scene Reconstruction Training**

运行

``` bash
cd scene_reconstruction
bash sim_render/cam/train_weather_scene.sh
```

脚本中`weather_type` 仅支持 "snowy", "rainy", "foggy"，且请确保你所选的`data_root`中存在`images_{weather_type}`，即已经通过Background Editing生成对应天气的图片。

请确保脚本`camera_ids`和在Background Editing选择的摄像头id一致。

------------------------------------------------------------------------

### **3. Render with Dynamic Particles**

首先编辑`edit_config.yaml`，编辑`weather_type`。`weather_type`为`“”`则渲染时不添加动态粒子，为`rainy`、`snowy`、`foggy`，则会在渲染时添加对应天气的动态粒子

然后运行：

``` bash
cd scene_construction
export PYTHONPATH=$(pwd)
python sim_render/cam/render_edit.sh
```

If you want to adjust the severity or the results aren't satisfying, feel free to tune the parameters in `configs/particle_config.yaml`

## Edit Dynamic Objects

First edit `edit_config.yaml`.

The system supports the following dynamic object editing operations:

- `trajectory_edit`: modify the trajectory of an existing rigid object using time-windowed offset segments.
- `add`: add a new rigid object from the asset library and generate its trajectory from waypoints.
- `remove`: delete existing rigid object instances from the scene.
- `replace`: replace an existing rigid object's appearance with a new asset while preserving its trajectory.

### Example

```yaml
legend:
  rigid: false
  smpl: false

weather_type: ""

Nodes:
  RigidNodes:
    trajectory_edit:
      instance_id: [35]
      offset_windows:
        - [[0, 20, [0, 3, 0]], [20, 40, [0, -3, 0]]]
    add:
      ref_id: [-1]
      add_obj:
        - "/nas_thoru/oldbak/lcy/asset/car_sample_van.ply"
      waypoints:
        - - [0, -3, -0.2]
          - [6, 0, -0.2]
          - [12, 0, -0.2]
      method: "cubic"
      time_window:
        - [20, 80]
      color_correct: true
      brightness_radius: 5.0
    remove:
      instance_id: [1]
    replace:
      target_id: [1]
      replace_obj:
        - "/path/to/asset.ply"
```

### `trajectory_edit`

Used to modify the trajectory of an existing rigid object by applying time-windowed offset segments. Each window specifies a frame range and a 3D delta; offsets are accumulated additively across windows and interpolated with smoothstep easing.

Parameters:

- `instance_id`: list of target rigid object IDs.
- `offset_windows`: list of time windows, where each window is `[start_frame, end_frame, [dx, dy, dz]]`.

Window behavior:

- If `start_frame == end_frame`: the delta is applied as a **constant global translation** to all frames.
- Otherwise: the delta is interpolated from 0 to full value across `[start_frame, end_frame)` using **smoothstep** easing, then held constant for all subsequent frames.
- Multiple windows **accumulate additively** — later windows stack on top of earlier ones.
- The object's heading (quaternion) is **automatically corrected** based on the new motion direction.

Example:

```yaml
trajectory_edit:
  instance_id: [35]
  offset_windows:
    - [[0, 20, [0, 3, 0]], [20, 40, [0, -3, 0]]]
```

This modifies the trajectory of instance `35`: from frame 0 to 20 it smoothly shifts laterally by `+3` in Y, then from frame 20 to 40 it smoothly shifts back by `-3` in Y.

---

### `add`

Used to add a new rigid object from the asset library.

Parameters:

- `add_obj`: path to the `.ply` asset.
- `ref_id`: reference trajectory source (see [Reference Trajectory](#reference-trajectory)).
- `waypoints`: offset trajectory relative to the reference trajectory (see [Waypoint Format](#waypoint-format) and [Trajectory Rules](#trajectory-rules)).
- `method`: interpolation method (see [Interpolation Methods](#interpolation-methods)).
- `time_window`: visble time window for new objects.
- `color_correct`: (optional) enable brightness matching against nearby scene Gaussians. Default: `false`.
- `brightness_radius`: (optional) 3D spatial query radius in meters for finding reference Gaussians during color correction. Default: `3.0`.

Example:

```yaml
add:
  ref_id: [-1]
  add_obj:
    - "/nas_thoru/oldbak/lcy/asset/car_sample_van.ply"
  waypoints:
    - - [0, -3, -0.2]
      - [6, 0, -0.2]
      - [12, 0, -0.2]
  method: "cubic"
  time_window:
    - [20, 80]
  color_correct: true
  brightness_radius: 5.0
```

```yaml
add:
  ref_id: [-1]
  add_obj: [0]
  waypoints:
    - - [0, -3, -0.2]
      - [6, 0, -0.2]
      - [12, 0, -0.2]
  method: "cubic"
  time_window:
    - [20, 80]
  color_correct: true
  brightness_radius: 5.0
```

---

### Reference Trajectory

The newly added object follows a reference trajectory specified by `ref_id`.

Use ego trajectory:

```yaml
ref_id: [-1]
```

Use an existing rigid object trajectory:

```yaml
ref_id: [35]
```

This means the new object follows the trajectory of instance `35`.

---

### Trajectory Rules

The waypoints define offset keyframes relative to the reference trajectory. The behavior depends on the number of waypoints:

- **1 waypoint**: the entire trajectory is translated by this constant offset (no interpolation).
- **2 waypoints**: linear interpolation is used to transition from the first offset to the second.
- **3 or more waypoints**: either `"linear"` or `"cubic"` interpolation can be used.

When `method: "cubic"` is requested with only 2 waypoints, it automatically degrades to `"linear"` with a warning.

---

### Waypoint Format

Each waypoint must be a 3D offset:

```yaml
[x, y, z]
```

Example:

```yaml
[0, -3, -0.2]
```

where:

- `x`: longitudinal offset (meters)
- `y`: lateral offset (meters)
- `z`: vertical offset (meters)

---

### Interpolation Methods

Linear interpolation:

```yaml
method: "linear"
```

Cubic spline interpolation:

```yaml
method: "cubic"
```

Requirements:

- `linear` requires at least 2 waypoints.
- `cubic` requires at least 3 waypoints.

When only 1 waypoint is provided, the trajectory is translated directly without interpolation.

---

### Notes

`method` accepts both a plain string (broadcast to all items) and a list (one per item).

Both are valid:

```yaml
method: "linear"
```

```yaml
method: ["cubic"]
```

When a single string is provided for multiple `add_obj` entries, it is automatically broadcast. When a list is provided, its length must match the number of items.

Additional constraints:

- All assets added through `add_obj` must be `.ply` files.
- Each waypoint must follow the `[x, y, z]` format.
- Newly added rigid objects automatically align their heading with the generated trajectory.
- `ref_id = -1` uses the ego trajectory as reference.
- `ref_id = instance_id` uses the specified rigid object's trajectory as reference.

---

### `remove`

Used to delete existing rigid object instances from the scene.

Parameters:

- `instance_id`: list of rigid object instance IDs to remove.

Example:

```yaml
remove:
  instance_id: [1, 5, 10]
```

---

### `replace`

Used to replace an existing rigid object's appearance with a new asset while keeping its trajectory.

Parameters:

- `target_id`: list of target rigid object instance IDs to replace.
- `replace_obj`: list of replacement sources. Can be:
  - A `.ply` file path (loads Gaussian appearance from an external asset).
  - An integer instance ID (copies Gaussian appearance from another instance in the scene).

Example:

```yaml
replace:
  target_id: [1, 2]
  replace_obj:
    - "/path/to/car_ambulance.ply"
    - 5
```

This replaces instance `1`'s Gaussians with the ambulance asset, and instance `2`'s Gaussians with a copy of instance `5`'s appearance.

## Edit Static Obstacles

First edit `edit_config.yaml`.

The following example adds static obstacles into the reconstructed scene.

### Example

```yaml
is_legend: false
weather_type: ""
debug: false

Nodes:
  Background:
    add:
      add_obj:
        - "refined_point_cloud.ply"
      offset:
        - [5, -2.1, -1.2]
      time_window:
        - [0, -1, 10]
      rotation: 
        - [0, 0, 90]
```

### `add_obj`

Path to the static obstacle asset.

Specify the 3D Gaussian Splatting (3DGS) or point cloud asset file to be inserted into the scene.

Example:

```yaml
add_obj:
  - "refined_point_cloud.ply"
```

---

### `offset`

Relative position of the static obstacle with respect to the ego vehicle.

Format:

```yaml
offset:
  - [x, y, z]
```

where:

- `x`: longitudinal offset (forward/backward direction)
- `y`: lateral offset (left/right direction)
- `z`: vertical offset (up/down direction)

Example:

```yaml
offset:
  - [5, -2.1, -1.2]
```

This places the obstacle:

- 5 m ahead of the ego vehicle
- 2.1 m to the right
- 1.2 m below

---

### `time_window`

Controls when and how frequently static obstacles are generated along the ego trajectory.

Format:

```yaml
time_window:
  - [start_frame, end_frame, step]
```

where:

- `start_frame`: obstacle generation starts from the ego vehicle position at this frame
- `end_frame`: obstacle generation ends at the ego vehicle position at this frame
- `step`: generate a new obstacle every `step` frames

Example:

```yaml
time_window:
  - [0, -1, 10]
```

This means:

- Start generating obstacles from frame 0.
- Continue until the end of the sequence (`-1`).
- Generate one obstacle every 10 frames.

---

### `rotation`
Initial rotation of the inserted obstacle.

Format:

```yaml
rotation:
  - [roll, pitch, yaw]
```

where:

- `roll`: rotation around the x-axis (degrees)
- `pitch`: rotation around the y-axis (degrees)
- `yaw`: rotation around the z-axis (degrees)

Example:

```yaml
rotation:
  - [0, 0, 90]
```

This rotates the obstacle by 90 degrees around the vertical axis.

---

## Pedestrian Editing 行人编辑

行人编辑通过 `configs/edit_config.yaml` 控制，主要配置 `Nodes.SMPLNodes.add`、`replace`、`remove`、`trajectory`。配置完成后运行 `sim_render/cam/render_edit.sh` 渲染编辑结果。

### **1. 查看 SMPL Instance IDs**

设置 `legend.smpl: true` 后运行渲染脚本，会导出 `legend_smpl.png`，用于确认场景中每个行人对应的 SMPL 实例 id。

```yaml
legend:
  rigid: false
  smpl: true
```

### **2. 导出 SMPL 资产**

修改 `sim_render/cam/export_smpl_assets.sh` 中的 `ckpt_path`、`output_dir`、`instance_ids`，然后运行：

```bash
bash sim_render/cam/export_smpl_assets.sh
```

运行后会在 `output_dir` 下生成 `smpl_instance_xxx.ply` 和 `smpl_instance_xxx_motion.pt`，可用于新增或替换行人。

其中 `.ply` 是人体高斯外观/template 资产；`*_motion.pt` 是动作文件，保存该 SMPL 实例的轨迹、姿态和可见帧信息，不包含人体外观。配置里的 `add_motion` 和 `replace_motion` 就是引用这个动作文件。

### **3. 配置行人编辑**

删除行人：当前是软删除，会保留实例 id，但该实例在所有帧中不可见。

```yaml
Nodes:
  SMPLNodes:
    remove:
      instance_id: [12]
```

轨迹平移：对整条轨迹做整体平移，不改变动作和可见帧。

```yaml
Nodes:
  SMPLNodes:
    trajectory:
      instance_id: [12]
      offset:
        - [0.0, 0.0, 1.0]
```

新增行人：`ref_id` 可以是场景中的 SMPL id，也可以是导出的 `*_motion.pt` 动作文件；`appear_frames` 表示从第几帧开始出现（当前约 10 帧/秒）；`offset` 是基于参考轨迹的 `[x, y, z]` 偏移。

```yaml
Nodes:
  SMPLNodes:
    add:
      ref_id: [12]
      add_obj: ["/path/to/smpl_instance_001.ply"]
      add_motion: ["/path/to/smpl_instance_001_motion.pt"]
      appear_frames: [30]
      offset:
        - [1.5, 0.0, 0.0]
```

替换行人：`replace_obj` 可以是 `.ply` 资产或场景中的 SMPL id；`replace_motion` 可以是 `*_motion.pt` 动作文件、场景中的 SMPL id，或 `null` 表示保留原动作。只替换动作时，`replace_obj` 写原 `target_id` 即可。

```yaml
Nodes:
  SMPLNodes:
    replace:
      target_id: [12]
      replace_obj: ["/path/to/new_person.ply"]
      replace_motion: [null]
```

批量编辑时，同一个操作里的所有列表长度必须一致，并按位置一一对应。

```yaml
add:
  ref_id: [12, "/path/to/reference_motion.pt"]
  add_obj:
    - "/path/to/person_A.ply"
    - "/path/to/person_B.ply"
  add_motion:
    - "/path/to/person_A_motion.pt"
    - "/path/to/person_B_motion.pt"
  appear_frames: [30, 45]
  offset:
    - [1.0, 0.0, 0.0]
    - [-1.0, 0.0, 0.0]
```

最后运行：

```bash
bash sim_render/cam/render_edit.sh
```


#### Notes

- Static obstacles do not have motion trajectories.
- Assets specified in `add_obj` should be 3DGS or point cloud files.
- Multiple obstacles can be generated automatically along the ego trajectory using `time_window`.
- The obstacle position at each generated frame is computed by applying the specified `offset` relative to the ego vehicle pose.

## 👏 Contributions
We're improving our project to develop a robust driving recom/sim system. Some areas we're focusing on:

- A real-time viewer for background and foreground visualization
- Scene editing and simulation tools
- Other Gaussian representations (e.g., 2DGS, surfels)

We welcome pull requests and collaborations. If you'd like to contribute or have questions, feel free to open an issue or contact [Ziyu Chen](https://github.com/ziyc) (ziyu.sjtu@gmail.com).

## 🙏 Acknowledgments
We utilize the rasterization kernel from [gsplat](https://github.com/nerfstudio-project/gsplat). Parts of our implementation are based on work from [EmerNeRF](https://github.com/NVlabs/EmerNeRF), [NerfStudio](https://github.com/nerfstudio-project/nerfstudio), [GART](https://github.com/JiahuiLei/GART), and [Neuralsim](https://github.com/PJLab-ADG/neuralsim). We've also implemented unofficial versions of [Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians), [PVG](https://github.com/fudan-zvg/PVG), and [Street Gaussians](https://github.com/zju3dv/street_gaussians), with reference to their original codebases.

We extend our deepest gratitude to the authors for their contributions to the community, which have greatly supported our research.

## Citation
```
@article{chen2024omnire,
    title={OmniRe: Omni Urban Scene Reconstruction},
    author={Chen, Ziyu and Yang, Jiawei and Huang, Jiahui and Lutio, Riccardo de and Esturo, Janick Martinez and Ivanovic, Boris and Litany, Or and Gojcic, Zan and Fidler, Sanja and Pavone, Marco and Song, Li and Wang, Yue},
    journal={arXiv preprint arXiv:2408.16760},
    year={2024}
}
```
