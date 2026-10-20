# H800 上的 Sparse4D 环境配置方法
## 第一步：准备权重文件
确保存在以下路径：
* `/data/Sparse4D/ckpt/sparse4dv3_r50.pth`
* `/data/Sparse4D/ckpt/resnet50-19c8e357.pth`
* `/data/Sparse4D/nuscenes_kmeans900.npy`

```sh
# 切换到 Sparse4D 目录下
cd third_party/Sparse4D
```

## 第二步：配置 conda 环境 （使用镜像请跳过这个步骤）

1. 安装 PyTorch 和 MMCV
```sh
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
```

2. 安装剩余依赖
```sh
# requirement.txt >>>
numpy==1.23.5
mmdet==2.28.2
urllib3==1.26.16
pyquaternion==0.9.9
nuscenes-devkit==1.1.10
yapf==0.33.0
tensorboard==2.14.0
motmetrics==1.1.3
pandas==1.1.5
imageio[ffmpeg]
# <<<

pip install -r requirement.txt
```

## 第三步：编译 deformable_aggregation CUDA 算子
```sh
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../../..  # 回到项目根目录
```
如果提示 CUDA 版本不匹配，通过下面的方法跳过检查：
```sh
vim /opt/conda/envs/sparse4d/lib/python3.9/site-packages/torch/utils/cpp_extension.py

# 文件内部
def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    return  # 增加这一行
 
    if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)
    ...    
```