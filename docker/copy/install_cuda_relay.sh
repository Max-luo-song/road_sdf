#!/bin/bash

set -e

export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
export CUDA_HOME=/usr/local/cuda

source /opt/conda/etc/profile.d/conda.sh
conda activate main

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install /opt/LiDAR-GS/submodules/simple-knn
pip install /opt/LiDAR-GS/submodules/diff_lidargs_rasterization
pip install /opt/LiDAR-GS/submodules/diff_lidargs_surfel_rasterization
pip install /opt/LiDAR-GS/extern/chamfer3D

echo "export http_proxy=http://172.26.255.17:3128" >> /etc/profile
echo "export https_proxy=http://172.26.255.17:3128" >> /etc/profile
echo "export ftp_proxy=http://172.26.255.17:3128" >> /etc/profile
