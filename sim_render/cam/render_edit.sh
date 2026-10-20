# 参数设置
################################################################################
ckpt_path="/nas/users/liuzy/data/scene_reconstruction_data/10clips/output/qcraft_20250818_140332_Q3707_990_1020_part01/20260324_lidar+cam0_1_2_3_5_6_9_10_12/checkpoint_final.pth"
edit_config_path="configs/edit_config.yaml"
output_path_postfix="_cone"
render_cam_ids="1"
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source scripts/utils.sh

if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi
echo "Using GPU: ${gpu}"
echo "Using checkpoint: $ckpt_path"

export PYTHONPATH=$(pwd)

# Build optional arguments
extra_args=""
if [ -n "${render_cam_ids}" ]; then
    extra_args="--render_cam_ids ${render_cam_ids}"
fi

python sim_render/cam/render_edit.py \
    --resume_from "${ckpt_path}" \
    --edit_config "${edit_config_path}" \
    --post_fix "${output_path_postfix}" \
    ${extra_args}
