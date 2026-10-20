# Parameters
################################################################################
ckpt_path="output/qcraft_20251105_152839_QCOYSD504206_1164_1179/20260120_lidar+cam1depth_loss/checkpoint_final.pth"
output_dir="output/qcraft_20251105_152839_QCOYSD504206_1164_1179/20260120_lidar+cam1depth_loss/smpl_assets"
instance_ids=(0)
alpha_thresh=0.001
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh

gpu=$(pick_gpu)
if [ -z "${gpu}" ]; then
    echo "no gpu found"
    exit 1
fi

export PYTHONPATH=$(pwd)

extra_args=""
if [ -n "$output_dir" ]; then
    extra_args="$extra_args --output_dir $output_dir"
fi

CUDA_VISIBLE_DEVICES=${gpu} python sim_render/cam/export_smpl_assets.py \
    --resume_from "$ckpt_path" \
    --instance_ids "${instance_ids[@]}" \
    --alpha_thresh "$alpha_thresh" \
    --export_ply \
    --export_motion \
    $extra_args
