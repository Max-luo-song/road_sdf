#!/bin/bash

# 对比两次训练输出的评估结果
# 用法: bash scripts/qcraft/compare.sh <log_dir1> <log_dir2>

################################################################################
# 参数设置
################################################################################

# 第一个训练结果的日志目录（旧结果）
# log_dir1="${1:-}"
log_dir1="output/qcraft_20251025_163358_QCOYSD504206_1595_1610/cam_0_1_2_3_5_6_7_9_10_11_12"

# 第二个训练结果的日志目录（新结果）
# log_dir2="${2:-}"
log_dir2="output/qcraft_20251025_163358_QCOYSD504206_1595_1610/cam_0_1_2_3_6_7_10_11_12_wo_depth_loss"

# wandb 配置
enable_wandb=false
wandb_project="scene_recon_compare_update"
wandb_entity="1zzhaozz-nanjing-university"
wandb_run_name="compare_$(date +%Y%m%d%H%M)"

# GPU 设置
gpu=-1

################################################################################

# 检查参数
if [ -z "$log_dir1" ] || [ -z "$log_dir2" ]; then
    echo "用法: bash scripts/qcraft/compare.sh <log_dir1> <log_dir2>"
    echo "示例: bash scripts/qcraft/compare.sh output/project1/run1 output/project1/run2"
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$log_dir1" ]; then
    echo "错误: 目录不存在: $log_dir1"
    exit 1
fi

if [ ! -d "$log_dir2" ]; then
    echo "错误: 目录不存在: $log_dir2"
    exit 1
fi

# 从 log_dir 中提取信息用于 wandb_run_name
if [ -z "$wandb_run_name" ]; then
    dir1_name=$(basename "$log_dir1")
    dir2_name=$(basename "$log_dir2")
    wandb_run_name="compare_${dir1_name}_vs_${dir2_name}"
fi

# 选择 GPU
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "警告: 未找到可用 GPU，将使用 CPU"
        gpu=""
    fi
fi

if [ -n "${gpu}" ]; then
    echo "使用 GPU: ${gpu}"
    export CUDA_VISIBLE_DEVICES=${gpu}
else
    echo "使用 CPU"
fi

# 设置 PYTHONPATH
export PYTHONPATH=$(pwd)

# 运行对比脚本
echo "开始对比训练结果..."
echo "旧结果目录: $log_dir1"
echo "新结果目录: $log_dir2"

python tools/compare.py \
    --log_dir1 "$log_dir1" \
    --log_dir2 "$log_dir2" \
    --num_cams 11 \
    $([ "$enable_wandb" = true ] && echo "--enable_wandb") \
    $([ -n "$wandb_project" ] && echo "--wandb_project" "$wandb_project") \
    $([ -n "$wandb_entity" ] && echo "--wandb_entity" "$wandb_entity") \
    $([ -n "$wandb_run_name" ] && echo "--wandb_run_name" "$wandb_run_name")

echo "对比完成！"

