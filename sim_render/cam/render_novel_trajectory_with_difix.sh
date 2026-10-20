#!/bin/bash

# DiFix3D 在线生成式修复 + 渐进式混合蒸馏总控脚本。
#
# 核心思路：
# 1. 从基础 3DGS checkpoint 渲染横向偏移的新轨迹图像。
# 2. 使用 DiFix3D 修复新轨迹图像，得到没有真实 GT 时使用的 pseudo-GT。
# 3. 用 L1 loss 将 pseudo-GT 蒸馏回 3DGS；穿插原轨迹 GT replay，减轻原视角退化。
# 4. 保存蒸馏后的 checkpoint；可选评测原轨迹和新轨迹。
#
# 轨迹偏移 token（novel_traj_shifts，逗号分隔）：
#   left_1  left_2  left_3   左移 1/2/3 米
#   right_1 right_2 right_3  右移 1/2/3 米
# 默认 6 条：left_1,left_2,left_3,right_1,right_2,right_3
# 每个新轨迹 stage 后自动插入 original_traj GT replay stage。
#
# 环境变量覆盖（pipeline 调用时使用）：
#   ckpt_path, experiment_output_dir, novel_traj_shifts, pipeline_gpu
#   difix_group=group0|group1  → 从 scripts/qcraft/common.sh 读 train_group*_views
#                              → 不传则不分组，使用全部默认相机
#   eval_original_traj_after_distill, eval_novel_traj_after_distill
#
# 蒸馏帧子采样：distill_frame_stride=3 表示保留第 0、3、6、... 帧。
# Progressive 模式总优化步数：
#   相机数 * 每个 stage 的蒸馏帧数 * sum(distill_stage_repeats)

################################################################################
# --- 可被 env 覆盖的核心参数 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ckpt_path=${ckpt_path:-"/nas/scenario_output/tuyuhang/scene_reconstruction/output/qcraft_20260601_151346_QCJPSD851972_2278_2298/20260609_lidar+cam0_1_2_3_5_6_7_9_10_11_12/checkpoint_final.pth"}
experiments=${experiments:-"output"}

novel_traj_shifts=${novel_traj_shifts:-"left_1,left_2,left_3"}
novel_distill_repeat=${novel_distill_repeat:-3}
original_distill_repeat=${original_distill_repeat:-6}

distill_ref_traj_as_original_stage=${distill_ref_traj_as_original_stage:-true}

# 分组：传 difix_group=group0|group1 时从 common.sh 读相机；不传则全相机。
difix_group=${difix_group:-"group0"}
if [ -n "${difix_group}" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/scripts/qcraft/common.sh"
    case "${difix_group}" in
        group0|0|train_group0)
            _group_views_csv="${train_group0_views}"
            ;;
        group1|1|train_group1)
            _group_views_csv="${train_group1_views}"
            ;;
        *)
            echo "ERROR: unknown difix_group='${difix_group}'. Use group0|group1, or leave empty for no grouping."
            exit 1
            ;;
    esac
    if [ -z "${_group_views_csv}" ]; then
        echo "ERROR: empty camera list for difix_group=${difix_group} in common.sh"
        exit 1
    fi
    IFS=',' read -r -a cam_ids <<< "${_group_views_csv}"
    distill_cam_ids=("${cam_ids[@]}")
    # 参考相机取该组第一路，避免默认 0 不在组内
    if [ -z "${distill_ref_cam_id:-}" ]; then
        distill_ref_cam_id="${cam_ids[0]}"
    fi
    echo "[INFO] difix_group=${difix_group} -> cams=${_group_views_csv}"
fi

if [ ${#cam_ids[@]} -eq 0 ]; then
    cam_ids=(0 1 2 3 5 6 7 9 10 11 12)
fi
if [ ${#distill_cam_ids[@]} -eq 0 ]; then
    distill_cam_ids=("${cam_ids[@]}")
fi
if [ ${#downscales[@]} -eq 0 ]; then
    downscales=()
    for ((_i = 0; _i < ${#cam_ids[@]}; _i++)); do
        downscales+=(1)
    done
fi

fps=${fps:-10}
render_rgb=${render_rgb:-true}
render_depth=${render_depth:-false}
save_images=${save_images:-true}
generate_lidar_pc=${generate_lidar_pc:-false}

# ------------------------- DiFix 蒸馏配置 -------------------------
enable_difix_distill=${enable_difix_distill:-true}

distill_ref_cam_id=${distill_ref_cam_id:-0}

distill_use_all_frames=${distill_use_all_frames:-true}
distill_max_frames=${distill_max_frames:--1}
distill_frame_stride=${distill_frame_stride:-3}

# 蒸馏优化器学习率 = distill_lr * distill_lr_scale。
distill_lr=${distill_lr:-1e-4}
distill_lr_scale=${distill_lr_scale:-1.0}

# DiFix3D 源码与预训练权重目录。
difix_src_dir=${difix_src_dir:-"/nas/oldbak/ga/code/Difix3D-main/src"}
difix_pretrained_dir=${difix_pretrained_dir:-"/nas/oldbak/ga/code/Difix3D-main/difix_ref"}
difix_device=${difix_device:-"cuda"}

# DiFix 推理参数。当前配置为单步修复，prompt 含义是“移除退化”。
difix_prompt=${difix_prompt:-"remove_degradation"}
difix_num_inference_steps=${difix_num_inference_steps:-1}
if [ ${#difix_timesteps[@]} -eq 0 ]; then
    difix_timesteps=(199)
fi
difix_guidance_scale=${difix_guidance_scale:-0.0}

# 使用同相机、同帧的原轨迹 GT 作为 DiFix ref_image，帮助修复结果保持场景一致。
difix_use_original_traj_ref=${difix_use_original_traj_ref:-true}
difix_ref_traj_type=${difix_ref_traj_type:-"original_traj"}

# 是否在蒸馏前导出原轨迹 GT 参考视频。关闭时要求目标目录中已存在 cam{id}.mp4。
generate_ref_videos=${generate_ref_videos:-true}
overwrite_ref_videos=${overwrite_ref_videos:-false}

# 仅当 DiFix 模型加载确实需要远程自定义代码时才开启。
difix_trust_remote_code=${difix_trust_remote_code:-false}

# 全 timeline 视频渲染策略：
# false = 每个 distill stage 结束后都渲染该 stage 对应轨迹；
# true（默认）= 所有 stage 蒸馏结束后，仅渲染原轨迹 + 横向偏移最大的新轨迹。
distill_render_max_offset_and_original=${distill_render_max_offset_and_original:-true}

# ------------------------- 蒸馏后评测配置 -------------------------
# 默认两者均关闭；需要评测时通过 env 显式开启。
eval_original_traj_after_distill=${eval_original_traj_after_distill:-false}
eval_novel_traj_after_distill=${eval_novel_traj_after_distill:-false}
eval_postfix=${eval_postfix:-"prog_mix_original_traj"}
eval_render_full=${eval_render_full:-true}
eval_render_test=${eval_render_test:-false}


# 新轨迹正式指标默认渲染全 timeline，而非仅评测 stride 子采样帧。
eval_novel_render_all_frames=${eval_novel_render_all_frames:-true}
skip_eval_novel_render_all_frames=${skip_eval_novel_render_all_frames:-false}
lpips_device=${lpips_device:-"cuda"}

# 最终 checkpoint 的完整路径。skip_final_fix_ckpt=true 时不会保存，也不会执行后续评测。

skip_final_fix_ckpt=${skip_final_fix_ckpt:-false}
pipeline_gpu=${pipeline_gpu:-""}
################################################################################

cd "${REPO_ROOT}"

source scripts/qcraft/difix_schedule.sh
build_difix_progressive_schedule \
    "${novel_traj_shifts}" \
    "${novel_distill_repeat}" \
    "${original_distill_repeat}" || exit 1

final_fix_ckpt_suffix=$(build_difix_ckpt_suffix \
    "${novel_traj_shifts}" \
    "${novel_distill_repeat}" \
    "${original_distill_repeat}") || exit 1
final_fix_ckpt_name="checkpoint_final_${final_fix_ckpt_suffix}.pth"
difix_output_suffix="${final_fix_ckpt_suffix}"

if [ -n "${experiment_output_dir:-}" ]; then
    : # 使用外部传入值（pipeline 场景）
else
    scene_name=$(basename "$(dirname "$(dirname "${ckpt_path}")")")
    experiment_output_dir="${experiments%/}/${scene_name}"
fi

difix_ref_video_dir=${difix_ref_video_dir:-"${experiment_output_dir}/refer_gt_video"}
novel_metrics_dir=${novel_metrics_dir:-"${experiment_output_dir}/metrics_novel_difix"}
final_fix_ckpt="${experiment_output_dir}/${final_fix_ckpt_name}"

echo "experiment_output_dir: ${experiment_output_dir}"
echo "difix_group: ${difix_group:-"(none, all cams)"}"
echo "novel_traj_shifts: ${novel_traj_shifts}"
echo "cam_ids: ${cam_ids[*]}"
echo "distill_cam_ids: ${distill_cam_ids[*]}"
echo "traj_types: ${traj_types[*]}"
echo "distill_stage_repeats: ${distill_stage_repeats[*]}"
echo "distill_render_max_offset_and_original: ${distill_render_max_offset_and_original}"
echo "final_fix_ckpt_suffix: ${final_fix_ckpt_suffix}"

source scripts/utils.sh
# 让 Python 能从当前仓库根目录导入项目模块。
export PYTHONPATH=$(pwd)

pipeline_start_ts=$(date +%s)

# 执行一个步骤、记录分钟级耗时，并在子命令失败时立即退出。
# 用法：run_timed_step "步骤名" 命令 参数...
run_timed_step() {
    local name="$1"
    shift
    local t0=$(date +%s)
    "$@"
    local rc=$?
    local msg="${name}: $(( ($(date +%s) - t0) / 60 )) min"
    echo "${msg}" | tee -a "${timing_log}"
    [ "$rc" -eq 0 ] || exit "$rc"
}

# 提前创建所有输出目录。
mkdir -p "${experiment_output_dir}" "${difix_ref_video_dir}"

# 所有主要步骤的耗时会追加写入该日志。
timing_log="${experiment_output_dir}/pipeline_timing.log"
{
    echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "ckpt_path: ${ckpt_path}"
    echo "experiment_output_dir: ${experiment_output_dir}"
    echo "novel_traj_shifts: ${novel_traj_shifts}"
} >> "${timing_log}"
echo "timing log: ${timing_log}"

# 评测脚本会从实验目录读取 config.yaml，因此将基础 checkpoint 同目录的配置复制过来。
src_config_path="$(dirname "$ckpt_path")/config.yaml"
dst_config_path="${experiment_output_dir}/config.yaml"
if [ -f "${src_config_path}" ]; then
    cp "${src_config_path}" "${dst_config_path}"
    echo "Copied config to experiment dir: ${dst_config_path}"
else
    echo "WARNING: source config not found: ${src_config_path}"
    echo "Eval may fail if ${dst_config_path} is missing."
fi

# 自动选择空闲 GPU，后续通过 CUDA_VISIBLE_DEVICES 限制进程只使用该卡。

if [ -n "${pipeline_gpu}" ]; then
    gpu="${pipeline_gpu}"
    echo "Using pipeline GPU: ${gpu}"
else
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
    echo "Using auto-picked GPU: ${gpu}"
fi

# 当 DiFix 使用原轨迹参考图时，先准备每个相机的 GT 参考视频。
if [ "$enable_difix_distill" = true ]; then
    if [ "$difix_use_original_traj_ref" = true ]; then
        if [ -z "$difix_ref_video_dir" ]; then
            echo "ERROR: difix_use_original_traj_ref=true, but difix_ref_video_dir is empty."
            exit 1
        fi
        if [ "$generate_ref_videos" = true ]; then
            # Bash 数组可以安全地逐项组装命令，避免相机列表参数被错误拆分。
            ref_cmd=(
                python sim_render/cam/export_reference_videos.py
                --resume_from "$ckpt_path"
                --cam_ids "${cam_ids[@]}"
                --downscales "${downscales[@]}"
                --fps "$fps"
                --output_dir "$difix_ref_video_dir"
            )
            [ "$overwrite_ref_videos" = true ] && ref_cmd+=(--overwrite)
            run_timed_step "export_reference_videos" \
                env CUDA_VISIBLE_DEVICES="${gpu}" "${ref_cmd[@]}"
        else
            # 不重新生成时，逐个检查参与蒸馏的相机是否已有参考视频。
            for cam_id in "${distill_cam_ids[@]}"; do
                if [ ! -f "${difix_ref_video_dir}/cam${cam_id}.mp4" ]; then
                    echo "ERROR: GT ref video not found: ${difix_ref_video_dir}/cam${cam_id}.mp4"
                    exit 1
                fi
            done
        fi
    fi
fi

# 组装主命令：render_novel_trajectory.py 会按 traj_types 顺序执行各 stage。
cmd=(
    python sim_render/cam/render_novel_trajectory.py
    --resume_from "$ckpt_path"
    --traj_types "${traj_types[@]}"
    --cam_ids "${cam_ids[@]}"
    --downscales "${downscales[@]}"
    --fps "$fps"
)

# 仅在对应布尔开关为 true 时追加 argparse 的 store_true 参数。
[ "$render_rgb" = true ] && cmd+=(--render_rgb)
[ "$render_depth" = true ] && cmd+=(--render_depth)
[ "$save_images" = true ] && cmd+=(--save_images)
[ "$generate_lidar_pc" = true ] && cmd+=(--generate_lidar_pc)

if [ "$enable_difix_distill" = true ]; then
    # 这些参数共同启用 Progressive DiFix 蒸馏：
    # - 新轨迹 stage：3DGS render -> DiFix fixed pseudo-GT -> L1 蒸馏
    # - 原轨迹 stage：原轨迹 GT -> L1 replay
    cmd+=(
        --enable_difix_distill
        --distill_ref_cam_id "$distill_ref_cam_id"
        --distill_cam_ids "${distill_cam_ids[@]}"
        --distill_use_all_frames
        --distill_max_frames "$distill_max_frames"
        --distill_frame_stride "$distill_frame_stride"
        --distill_lr "$distill_lr"
        --distill_lr_scale "$distill_lr_scale"
        --final_fix_ckpt "$final_fix_ckpt_name"
        --difix_src_dir "$difix_src_dir"
        --difix_pretrained_dir "$difix_pretrained_dir"
        --difix_device "$difix_device"
        --difix_prompt "$difix_prompt"
        --difix_num_inference_steps "$difix_num_inference_steps"
        --difix_timesteps "${difix_timesteps[@]}"
        --difix_guidance_scale "$difix_guidance_scale"
        --difix_output_suffix="$difix_output_suffix"
        --difix_ref_traj_type "$difix_ref_traj_type"
        --difix_ref_video_dir "$difix_ref_video_dir"
    )
    # 非空 repeats 数组会启用按 stage 确定性训练的 Progressive 模式。
    if [ ${#distill_stage_repeats[@]} -gt 0 ]; then
        cmd+=(--distill_stage_repeats "${distill_stage_repeats[@]}")
    fi
    [ "$distill_ref_traj_as_original_stage" = true ] && cmd+=(--distill_ref_traj_as_original_stage)
    [ "$difix_use_original_traj_ref" = true ] && cmd+=(--difix_use_original_traj_ref)
    [ "$difix_trust_remote_code" = true ] && cmd+=(--difix_trust_remote_code)
    [ "$skip_final_fix_ckpt" = true ] && cmd+=(--skip_final_fix_ckpt)
    [ "$distill_render_max_offset_and_original" = true ] && cmd+=(--distill_render_max_offset_and_original)
fi
# opts 风格配置：将本次实验输出目录传给项目配置系统的 log_dir。
cmd+=("log_dir=${experiment_output_dir}")

# 执行“新轨迹渲染 + DiFix 修复 + 蒸馏”。
run_timed_step "render_novel_trajectory_distill" \
    env CUDA_VISIBLE_DEVICES="${gpu}" PIPELINE_TIMING_LOG="${timing_log}" "${cmd[@]}"

# 确定后续评测使用的蒸馏后 checkpoint，并在缺失时尽早报错。
final_ckpt_path=""
if [ "$enable_difix_distill" = true ] && [ "$skip_final_fix_ckpt" = false ]; then
    final_ckpt_path="${final_fix_ckpt}"
    if [ ! -f "${final_ckpt_path}" ]; then
        echo "ERROR: final checkpoint not found: ${final_ckpt_path}"
        exit 1
    fi
fi

# 原轨迹评测：使用真实 GT，检查生成式修复蒸馏是否损伤已知视角质量。
if [ "$eval_original_traj_after_distill" = true ] && [ -n "${final_ckpt_path}" ]; then
    run_timed_step "eval_original_traj" \
        env \
        gpu="${gpu}" \
        ckpt_path="${final_ckpt_path}" \
        eval_log_dir="${experiment_output_dir}" \
        eval_postfix="${eval_postfix}" \
        render_full="${eval_render_full}" \
        render_test="${eval_render_test}" \
        bash scripts/qcraft/cal_new_traj_pref.sh
fi

# 新轨迹评测：
# - ckpt_path：蒸馏后的模型，用于生成 after。
# - ckpt_path_before：基础模型，用于生成 before，并由 DiFix 得到 fixed pseudo-GT。
# 指标主要比较 before/fixed/after，以判断蒸馏是否吸收了 DiFix 的修复效果。
if [ "$eval_novel_traj_after_distill" = true ] && [ -n "${final_ckpt_path}" ]; then
    distill_base_ckpt_path="${ckpt_path}"
    run_timed_step "eval_novel_traj_pseudo_gt" \
        env \
        gpu="${gpu}" \
        experiment_dir="${experiment_output_dir}" \
        ckpt_path="${final_ckpt_path}" \
        ckpt_path_before="${distill_base_ckpt_path}" \
        render_video_postfix="${eval_postfix}" \
        render_full="${eval_render_full}" \
        render_test="${eval_render_test}" \
        novel_metrics_dir="${novel_metrics_dir}" \
        render_all_frames="${eval_novel_render_all_frames}" \
        skip_render_all_frames="${skip_eval_novel_render_all_frames}" \
        lpips_device="${lpips_device}" \
        difix_ref_video_dir="${difix_ref_video_dir}" \
        difix_use_original_traj_ref="${difix_use_original_traj_ref}" \
        difix_src_dir="${difix_src_dir}" \
        difix_pretrained_dir="${difix_pretrained_dir}" \
        bash scripts/qcraft/eval_novel_with_difix_gt.sh
fi

# 汇总整条 pipeline 的总耗时。
pipeline_elapsed=$(( ($(date +%s) - pipeline_start_ts) / 60 ))
echo "total: ${pipeline_elapsed} min" | tee -a "${timing_log}"
echo "end: $(date '+%Y-%m-%d %H:%M:%S')" >> "${timing_log}"
