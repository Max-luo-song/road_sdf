# 参数设置
################################################################################
scene_id="clip_1746752396800"
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh

export PYTHONPATH=$(pwd)

target_dir="data/chery/processed" && mkdir -p "${target_dir}"
source /opt/conda/etc/profile.d/conda.sh && conda activate main

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
  echo "no gpu found"
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpu} python datasets/preprocess.py \
    --data_root data/chery/raw/ \
    --target_dir ${target_dir} \
    --dataset chery \
    --split training \
    --scene_ids $scene_id \
    --workers 2 \
    --process_keys images lidar calib pose dynamic_masks objects lidar_velocities