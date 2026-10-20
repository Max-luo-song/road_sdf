# 参数设置
################################################################################
gpu=-1
data_root="data/qcraft/raw" # 原始数据目录
scene_list_path=data/qcraft_scenes.txt
################################################################################

# Pick an avaliable gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
    fi
fi
echo "Using GPU: ${gpu}"

# 激活 Conda 环境
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate sparse4d

# 获取绝对路径
abs_data_root=$(realpath "${data_root}")

cd third_party/Sparse4D
export CUDA_VISIBLE_DEVICES=${gpu}

python infer_qcraft.py \
    --data_root ${abs_data_root} \
    --scene_list_path ${scene_list_path}

cd ../../