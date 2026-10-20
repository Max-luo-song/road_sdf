#!/bin/bash
set -euo pipefail

# =========================
# 默认配置（支持 bash 前缀参数覆盖）
# 例如：
# experiment_dir=/path/to/exp \
# ckpt_path=/path/to/final.pth \
# ckpt_path_before=/path/to/base.pth \
# bash scripts/qcraft/eval_novel_with_difix_gt.sh
# =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_repo_root="$(cd "${SCRIPT_DIR}/../.." && pwd)"

repo_root=${repo_root:-"${default_repo_root}"}
experiment_dir=${experiment_dir:-"/nas/oldbak/ga/code/scene_reconstruction/output_scene_data_step_left_1.5_radio_6"}
ckpt_path=${ckpt_path:-"${experiment_dir}/checkpoint_final_difix_prog_left1p5_ori_1.5_radio_6.pth"}
ckpt_path_before=${ckpt_path_before:-""}

gpu=${gpu:-"0"}
render_video_postfix=${render_video_postfix:-"prog_mix_original_traj"}
render_full=${render_full:-"true"}
render_test=${render_test:-"false"}

python_bin=${python_bin:-"python"}
lpips_device=${lpips_device:-"cuda"}
novel_metrics_dir=${novel_metrics_dir:-"${experiment_dir}/metrics_novel_difix"}

# 新轨迹全帧评测（与蒸馏 stride 子采样解耦）
render_all_frames=${render_all_frames:-"true"}
skip_render_all_frames=${skip_render_all_frames:-"false"}
difix_ref_video_dir=${difix_ref_video_dir:-"${experiment_dir}/refer_gt_video"}
difix_use_original_traj_ref=${difix_use_original_traj_ref:-"true"}
difix_src_dir=${difix_src_dir:-"/nas/oldbak/ga/code/Difix3D-main/src"}
difix_pretrained_dir=${difix_pretrained_dir:-"/nas/oldbak/ga/code/Difix3D-main/difix_ref"}

# =========================
# 基础检查
# =========================
if [ ! -d "${repo_root}" ]; then
  echo "ERROR: repo_root not found: ${repo_root}"
  exit 1
fi
if [ ! -d "${experiment_dir}" ]; then
  echo "ERROR: experiment_dir not found: ${experiment_dir}"
  exit 1
fi
if [ ! -f "${ckpt_path}" ]; then
  echo "ERROR: checkpoint not found: ${ckpt_path}"
  exit 1
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

if [ "${render_all_frames}" = "true" ] && [ "${skip_render_all_frames}" != "true" ]; then
  if [ -z "${ckpt_path_before}" ]; then
    echo "ERROR: ckpt_path_before is required for full-frame novel eval."
    echo "Set ckpt_path_before=/path/to/checkpoint_final.pth (pre-distill base checkpoint)."
    exit 1
  fi
  if [ ! -f "${ckpt_path_before}" ]; then
    echo "ERROR: ckpt_path_before not found: ${ckpt_path_before}"
    exit 1
  fi
fi

echo "==================== [1/2] 原轨迹 eval（全帧） ===================="
echo "Experiment dir: ${experiment_dir}"
echo "Checkpoint: ${ckpt_path}"
echo "GPU: ${gpu}"

# CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" tools/eval.py \
#   --resume_from "${ckpt_path}" \
#   --render_video_postfix "${render_video_postfix}" \
#   "log_dir=${experiment_dir}" \
#   "render.render_full=${render_full}" \
#   "render.render_test=${render_test}"

echo "==================== [2/2] 新轨迹 pseudo-GT eval（全帧渲染） ===================="
echo "Novel metrics dir: ${novel_metrics_dir}"
echo "LPIPS device: ${lpips_device}"
echo "render_all_frames: ${render_all_frames}"
echo "ckpt_path (after): ${ckpt_path}"
echo "ckpt_path_before: ${ckpt_path_before}"

eval_cmd=(
  "${python_bin}" tools/eval_novel_with_difix_gt.py
  --experiment_dir "${experiment_dir}"
  --output_dir "${novel_metrics_dir}"
  --lpips_device "${lpips_device}"
  --ckpt_path "${ckpt_path}"
  --ckpt_path_before "${ckpt_path_before}"
  --difix_src_dir "${difix_src_dir}"
  --difix_pretrained_dir "${difix_pretrained_dir}"
  --difix_ref_video_dir "${difix_ref_video_dir}"
)

if [ "${render_all_frames}" = "true" ] && [ "${skip_render_all_frames}" != "true" ]; then
  eval_cmd+=(--render_all_frames)
else
  eval_cmd+=(--skip_render_all_frames)
fi
if [ "${difix_use_original_traj_ref}" = "true" ]; then
  eval_cmd+=(--difix_use_original_traj_ref)
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${eval_cmd[@]}"

echo "==================== DONE ===================="
echo "原轨迹指标目录: ${experiment_dir}/metrics_eval"
echo "新轨迹指标目录: ${novel_metrics_dir}"
echo "新轨迹全帧 PNG 目录: ${experiment_dir}/difix_eval_all_frames_*"
