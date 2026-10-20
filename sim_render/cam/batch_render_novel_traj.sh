#!/bin/bash

# 参数设置
################################################################################
# 1. 基础路径设置
scenes_file="data/qcraft_scenes.txt"
# 训练输出的总目录，脚本会去这里寻找 checkpoint
checkpoint_root="output" 
# checkpoint 的子目录匹配模式 (根据你的训练脚本命名规则)
# 如果你的训练输出目录名包含特定后缀，可以在这里设置

# 2. 渲染参数 (保持和你原始渲染脚本一致)
traj_types=(
    original_traj
    left_shift_1m
    left_shift_3m
    right_shift_1m
    right_shift_3m
)

cam_ids=(0 1 2 3 5 6 7 9 10 11 12)
downscales=(1 1 1 1 1 1 1 1 1 1 1)
fps=10

render_rgb=true
render_depth=false
save_images=true
generate_lidar_pc=false

# 3. 其他设置
gpu=-1 # -1 表示自动选择
stop_on_error=false
################################################################################

source scripts/utils.sh
export PYTHONPATH=$(pwd)

# 日志函数
log_info() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }
log_success() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS] $1"; }

# 从文件读取 scene 列表 (复用训练脚本的逻辑)
read_scenes_from_file() {
    local file=$1
    local -n scene_ids_ref=$2
    if [ ! -f "$file" ]; then log_error "文件不存在: $file"; return 1; fi
    
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        IFS=',' read -r scene_id _ <<< "$line" # 只取第一个字段 scene_id
        scene_id=$(echo "$scene_id" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [ -n "$scene_id" ] && scene_ids_ref+=("$scene_id")
    done < "$file"
    log_info "共读取到 ${#scene_ids_ref[@]} 个 scene"
}

# 渲染单个场景
render_scene() {
    local scene_id=$1
    log_info "------------------------------------------"
    log_info "正在处理场景: $scene_id"

    # 自动寻找 checkpoint 路径
    # 搜索模式: checkpoint_root/qcraft_scene_id/日期_后缀/checkpoint_final.pth
    # 使用 find 寻找该场景文件夹下最新的 checkpoint_final.pth
    local scene_dir="${checkpoint_root}/qcraft_${scene_id}"
    
    if [ ! -d "$scene_dir" ]; then
        log_error "找不到场景目录: $scene_dir，跳过。"
        return 1
    fi

    # 寻找 checkpoint_final.pth 文件
    local ckpt_path=$(find "$scene_dir" -name "checkpoint_final.pth" | head -n 1)

    if [ -z "$ckpt_path" ]; then
        log_error "在 $scene_dir 中找不到 checkpoint_final.pth，跳过。"
        return 1
    fi

    log_info "找到 Checkpoint: $ckpt_path"

    # 组装布尔参数
    local bool_args=""
    [ "$render_rgb" = true ] && bool_args="$bool_args --render_rgb"
    [ "$render_depth" = true ] && bool_args="$bool_args --render_depth"
    [ "$save_images" = true ] && bool_args="$bool_args --save_images"
    [ "$generate_lidar_pc" = true ] && bool_args="$bool_args --generate_lidar_pc"

    # 执行渲染命令
    CUDA_VISIBLE_DEVICES=${gpu} python sim_render/cam/render_novel_trajectory.py \
        --resume_from "$ckpt_path" \
        --traj_types "${traj_types[@]}" \
        --cam_ids "${cam_ids[@]}" \
        --downscales "${downscales[@]}" \
        --fps $fps \
        $bool_args

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_success "场景 $scene_id 渲染完成"
    else
        log_error "场景 $scene_id 渲染失败 (退出码: $exit_code)"
        return 1
    fi
}

# 主程序
main() {
    local scene_ids=()
    read_scenes_from_file "$scenes_file" scene_ids

    # GPU 选择
    if [ "${gpu}" = "-1" ]; then
        gpu=$(pick_gpu)
        if [ -z "${gpu}" ]; then log_error "未找到可用 GPU"; exit 1; fi
    fi
    log_info "使用 GPU: ${gpu}"

    local total=${#scene_ids[@]}
    local count=0

    for i in "${!scene_ids[@]}"; do
        ((count++))
        log_info "进度: $count/$total"
        
        if ! render_scene "${scene_ids[$i]}"; then
            if [ "$stop_on_error" = true ]; then
                log_error "发生错误，停止后续任务。"
                exit 1
            fi
        fi
    done

    log_success "所有批量渲染任务执行完毕！"
}

main "$@"