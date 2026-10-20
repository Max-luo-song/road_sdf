#!/bin/bash
set -euo pipefail

################################################################################
# Evaluate the new DiFix checkpoint on the original trajectory.
#
# It renders all configured camera views on the full/original trajectory, saves
# rendered images and GT images, and computes PSNR / SSIM / LPIPS through
# tools/eval.py.
################################################################################

gpu=${gpu:--1}
ckpt_path=${ckpt_path:-"output/qcraft_20250818_152739_Q3720_100_130_part01/20260506_lidar+cam0_1_2_3_5_6_7_9_10_11_12baseline/checkpoint_final_difix_all_left_offset05.pth"}

# tools/eval.py requires config.yaml next to the checkpoint.  The DiFix-only
# output directory may contain only the final .pth, so use the matching original
# training config as a fallback.
config_path=${config_path:-""}
eval_postfix=${eval_postfix:-"new_weight_original_traj"}
render_full=${render_full:-true}
render_test=${render_test:-false}
conda_env=${conda_env:-"main"}

# Optional overrides. Leave empty to use config.yaml values.
camera_ids=${camera_ids:-""}
downscales=${downscales:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source scripts/utils.sh

if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi

if [ ! -f "${ckpt_path}" ]; then
    echo "Checkpoint not found: ${ckpt_path}"
    exit 1
fi

ckpt_dir="$(dirname "${ckpt_path}")"
target_config="${ckpt_dir}/config.yaml"
eval_log_dir=${eval_log_dir:-"${ckpt_dir}"}

if [ -n "${config_path}" ]; then
    if [ ! -f "${config_path}" ]; then
        echo "config_path not found: ${config_path}"
        exit 1
    fi
    if [ "${config_path}" != "${target_config}" ]; then
        cp "${config_path}" "${target_config}"
    fi
elif [ ! -f "${target_config}" ]; then
    main_repo_root="${REPO_ROOT/scene_reconstruction-ga/scene_reconstruction-main}"
    fallback_config="${main_repo_root}/${target_config}"
    if [ -f "${fallback_config}" ]; then
        cp "${fallback_config}" "${target_config}"
        echo "Copied fallback config: ${fallback_config} -> ${target_config}"
    else
        echo "Missing ${target_config}."
        echo "Set config_path=/path/to/config.yaml when running this script."
        exit 1
    fi
fi

opts=(
    "log_dir=${eval_log_dir}"
    "render.render_full=${render_full}"
    "render.render_test=${render_test}"
)

if [ -n "${camera_ids}" ]; then
    opts+=("data.pixel_source.cameras=[${camera_ids}]")
fi

if [ -n "${downscales}" ]; then
    opts+=("data.pixel_source.downscale_when_loading=[${downscales}]")
fi

echo "Using GPU: ${gpu}"
echo "Using checkpoint: ${ckpt_path}"
echo "Using config: ${target_config}"
echo "Eval log_dir: ${eval_log_dir}"
echo "Eval postfix: ${eval_postfix}"

videos_eval_dir="${eval_log_dir}/videos_eval"
if [ -d "${videos_eval_dir}" ]; then
    stale_layout_mp4_backup="${videos_eval_dir}/_mp4_backup_$(date +%Y%m%d_%H%M%S)"
    moved_stale_layout_mp4=false
    while IFS= read -r -d '' stale_mp4; do
        if [ "${moved_stale_layout_mp4}" = false ]; then
            mkdir -p "${stale_layout_mp4_backup}"
            moved_stale_layout_mp4=true
        fi
        mv "${stale_mp4}" "${stale_layout_mp4_backup}/"
    done < <(find "${videos_eval_dir}" -maxdepth 1 -type f -name '*layout*.mp4' -print0)
    if [ "${moved_stale_layout_mp4}" = true ]; then
        echo "Moved stale layout mp4 files to: ${stale_layout_mp4_backup}"
    fi
fi

if [ -n "${conda_env}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "${conda_env}" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${conda_env}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=${gpu} python tools/eval.py \
    --resume_from "${ckpt_path}" \
    --render_video_postfix "${eval_postfix}" \
    "${opts[@]}"

latest_metrics=$(ls -t "${ckpt_dir}"/metrics_eval/images_full_*.json 2>/dev/null | head -n 1 || true)
if [ -n "${latest_metrics}" ]; then
    echo "Latest full-set metrics: ${latest_metrics}"
else
    echo "No full-set metrics json found under ${ckpt_dir}/metrics_eval"
fi

echo "Rendered images are under ${ckpt_dir}/videos_eval/full_set_*_${eval_postfix}_rgbs"
echo "GT images are under ${ckpt_dir}/videos_eval/full_set_*_${eval_postfix}_gt_rgbs"
