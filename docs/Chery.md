# 准备奇瑞数据集
基于 Waymo 数据集版本修改

## 1. 设置数据集目录
```shell
# Create the data directory or create a symbolic link to the data directory
mkdir -p ./data/chery
mkdir -p ./data/chery/processed 

ln -s $PATH_TO_CHERY ./data/chery/raw
```

## 2. 数据预处理
#### 运行预处理脚本
`data/chery_scenes.txt` 存放了需要预处理的场景，运行 `scripts/chery/preprocess_data.sh` 进行预处理，预处理后的数据会存放在 `data/chery/processed` 目录下。

## 3. 提取 Mask
为了生成以下数据:
- **sky masks** 
- fine dynamic masks (可选)

遵循以下步骤：

#### 安装 `SegFormer`

:警告：SegFormer 依赖于 `mmcv-full=1.2.7`, 其依赖于 `pytorch=1.8` (pytorch<1.9)。因此，需要创建另一个 conda 环境。

```shell
#-- Set conda env
conda create -n segformer python=3.8
conda activate segformer
# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#-- Install mmcv-full
pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf
pip install mmcv-full==1.2.7 --no-cache-dir

#-- Clone and install segformer
cd third_party
git clone https://github.com/NVlabs/SegFormer
cd SegFormer
pip install .
```

通过 https://github.com/NVlabs/SegFormer#evaluation 的 Google Drive / One Drive 链接下载预训练模型 `segformer.b5.1024x1024.city.160k.pth`。

在 SegFormer 项目目录下创建 `pretrained` 文件夹，放入预训练模型权重文件。

#### 运行 Mask 提取脚本

回到本项目根目录，运行 `scripts/chery/extract_masks.sh`

## 4. Human Body Pose Processing

#### Prerequisites
To utilize the SMPL-Gaussian to model pedestrians, please first download the SMPL models.

1. Download SMPL v1.1 (`SMPL_python_v.1.1.0.zip`) from the [SMPL official website](https://smpl.is.tue.mpg.de/download.php)
2. Move `SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to `PROJECT_ROOT/smpl_models/SMPL_NEUTRAL.pkl`

SMPL-Nodes (SMPL-Gaussian Representation) requires Human Body Pose Sequences of pedestrians. We've developed a human body pose processing pipeline for in-the-wild driving data to generate this information. There are two ways to obtain these data:

####  Run the Extraction Pipeline
To process human body poses, follow the instructions in our [Human Pose Processing Guide](./HumanPose.md).

## 5. Data Structure
After completing all preprocessing steps, the project files should be organized according to the following structure:
```shell
ProjectPath/data/
  └── chery/
    ├── raw/
    │    ├── clip_1746752396800
    │    └── ...
    └── processed/
         └──training/
                ├── 001/
                │  ├──images/             # Images: {timestep:03d}_{cam_id}.jpg
                │  ├──lidar/              # LiDAR data: {timestep:03d}.bin
                │  ├──lidar_pose/         # LiDAR poses: {timestep:03d}.txt
                │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
                │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
                │  ├──sky_masks/          # Sky masks: {timestep:03d}_{cam_id}.png
                │  ├──dynamic_masks/      # Coarse dynamic masks: category/{timestep:03d}_{cam_id}.png
                │  ├──fine_dynamic_masks/ # (暂时没有) Fine dynamic masks: category/{timestep:03d}_{cam_id}.png
                │  ├──instances/          # Instances' bounding boxes information
                │  └──humanpose/          # （暂时没有）Preprocessed human body pose: smpl.pkl
                ├── 002/
                ├── ...
```