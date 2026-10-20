#!/bin/bash

################################################################################
# Batch Training Script - 支持多卡并行执行
# 功能：批量执行train.sh，每个场景训练前自动选择占用最小的GPU，支持多场景并行
################################################################################

# 定义要训练的场景列表
scenes=(
    "20250702_133223_Q2517_60_75"
    "20250811_152823_Q3706_1780_1790"
    # "20251025_163358_QCOYSD504206_1595_1610"
    "20251105_152839_QCOYSD504206_1164_1179"
    "20251203_095105_QCOYSD968166_893_908"
)

# 配置参数（可选，如果不设置则使用train.sh中的默认值）
camera_ids_config="0 1 2 3 5 6 7 9 10 11 12"  # 相机ID列表
lidar_type_config="lidar"  # lidar/visual
start_timestep_config=0
end_timestep_config=-1

# 并发控制参数 - 自动检测GPU数量
gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ $gpu_count -eq 0 ]; then
    echo "警告：未检测到GPU，设置最大并行数为1"
    max_parallel_jobs=1
else
    max_parallel_jobs=$gpu_count
    echo "检测到 $gpu_count 张GPU，设置最大并行数为 $max_parallel_jobs"
fi

# 任务管理数组
declare -A job_pids      # 存储任务PID
declare -A job_scenes    # 存储任务对应的场景名
declare -A job_gpus      # 存储任务使用的GPU
declare -A gpu_usage     # 记录每个GPU当前运行的任务数

# 获取占用最小的GPU函数（考虑已分配的任务）
function get_min_gpu() {
    # 获取所有GPU的内存使用情况
    local gpu_mem_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '{print $1, $2}')
    
    local best_gpu=""
    local min_score=999999999
    
    # 遍历所有GPU，计算综合评分（内存占用 + 已分配任务数 * 权重）
    while IFS= read -r line; do
        local gpu_id=$(echo $line | awk '{print $1}')
        local mem_used=$(echo $line | awk '{print $2}')
        local running_tasks=${gpu_usage[$gpu_id]:-0}
        
        # 综合评分：内存占用 + 运行任务数 * 10000（给任务数更高权重）
        local score=$((mem_used + running_tasks * 10000))
        
        if [ $score -lt $min_score ]; then
            min_score=$score
            best_gpu=$gpu_id
        fi
    done <<< "$gpu_mem_info"
    
    echo "$best_gpu"
}

# 等待直到有空闲的任务槽位
function wait_for_slot() {
    while true; do
        # 检查正在运行的任务数
        running_jobs=0
        for pid in "${!job_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running_jobs++))
            else
                # 任务已完成，清理
                wait "$pid"
                exit_code=$?
                scene="${job_scenes[$pid]}"
                gpu="${job_gpus[$pid]}"
                
                if [ $exit_code -eq 0 ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 场景 $scene (GPU:$gpu, PID:$pid) 训练完成"
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 场景 $scene (GPU:$gpu, PID:$pid) 训练失败 (exit code: $exit_code)"
                fi
                
                unset job_pids[$pid]
                unset job_scenes[$pid]
                
                # 减少该GPU的任务计数
                if [ ! -z "$gpu" ]; then
                    gpu_usage[$gpu]=$((${gpu_usage[$gpu]:-1} - 1))
                    if [ ${gpu_usage[$gpu]} -lt 0 ]; then
                        gpu_usage[$gpu]=0
                    fi
                fi
                unset job_gpus[$pid]
            fi
        done
        
        # 如果有空闲槽位或不限制并发数，则返回
        if [ $max_parallel_jobs -eq -1 ] || [ $running_jobs -lt $max_parallel_jobs ]; then
            break
        fi
        
        # 等待1秒后重新检查
        sleep 1
    done
}

# 等待所有任务完成
function wait_all_jobs() {
    echo ""
    echo "========================================"
    echo "等待所有训练任务完成..."
    echo "========================================"
    
    for pid in "${!job_pids[@]}"; do
        scene="${job_scenes[$pid]}"
        gpu="${job_gpus[$pid]}"
        echo "等待场景 $scene (GPU:$gpu, PID:$pid)..."
        
        wait "$pid"
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 场景 $scene (GPU:$gpu, PID:$pid) 训练完成"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 场景 $scene (GPU:$gpu, PID:$pid) 训练失败 (exit code: $exit_code)"
        fi
    done
    
    echo ""
    echo "所有训练任务已完成！"
}

# 主循环：遍历所有场景进行训练
echo "========================================"
echo "批量训练配置："
echo "- 场景总数: ${#scenes[@]}"
echo "- 最大并行数: $([ $max_parallel_jobs -eq -1 ] && echo '不限制' || echo $max_parallel_jobs)"
echo "========================================"
echo ""

for scene_idx in "${scenes[@]}"; do
    # 等待有空闲的任务槽位
    wait_for_slot
    
    echo "----------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 准备启动场景: $scene_idx"
    
    # 查找当前占用最小的GPU
    gpu=$(get_min_gpu)
    
    if [ -z "${gpu}" ]; then
        echo "错误：没有找到可用的GPU，跳过场景 $scene_idx"
        continue
    fi
    
    echo "为场景 $scene_idx 分配GPU: $gpu"
    
    # 创建临时训练脚本（基于train.sh修改）
    temp_train_script="scripts/qcraft/temp_train_${scene_idx}_$(date +%s).sh"
    cp scripts/qcraft/train.sh "$temp_train_script"
    
    # 修改临时脚本中的参数
    sed -i "s/^gpu=.*/gpu=${gpu}/" "$temp_train_script"
    sed -i "s/^scene_idx=.*/scene_idx=\"${scene_idx}\"/" "$temp_train_script"
    
    # 如果需要，可以修改其他参数
    if [ ! -z "$camera_ids_config" ]; then
        sed -i "s/^camera_ids=.*/camera_ids=(${camera_ids_config})/" "$temp_train_script"
    fi
    if [ ! -z "$lidar_type_config" ]; then
        sed -i "s/^lidar_type=.*/lidar_type=\"${lidar_type_config}\"/" "$temp_train_script"
    fi
    if [ ! -z "$start_timestep_config" ]; then
        sed -i "s/^start_timestep=.*/start_timestep=${start_timestep_config}/" "$temp_train_script"
    fi
    if [ ! -z "$end_timestep_config" ]; then
        sed -i "s/^end_timestep=.*/end_timestep=${end_timestep_config}/" "$temp_train_script"
    fi
    
    # 在临时脚本末尾添加清理命令
    echo "rm -f \"$temp_train_script\"" >> "$temp_train_script"
    
    # 4. 创建输出目录（确保日志文件路径存在）
    output_dir="output/qcraft_${scene_idx}"
    mkdir -p "$output_dir"
    
    # 后台执行训练
    CUDA_VISIBLE_DEVICES=${gpu} bash "$temp_train_script" > "output/qcraft_${scene_idx}/training.log" 2>&1 &
    pid=$!
    
    # 记录任务信息
    job_pids[$pid]=1
    job_scenes[$pid]="$scene_idx"
    job_gpus[$pid]="$gpu"
    
    # 增加该GPU的任务计数
    gpu_usage[$gpu]=$((${gpu_usage[$gpu]:-0} + 1))
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 场景 $scene_idx 已启动 (GPU:$gpu, PID:$pid, GPU任务数:${gpu_usage[$gpu]})"
    echo "----------------------------------------"
    echo ""
    
    # 短暂延迟，避免GPU选择冲突
    sleep 3
done

# 等待所有任务完成
wait_all_jobs
