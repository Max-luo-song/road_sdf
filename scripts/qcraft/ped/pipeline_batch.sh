# 参数设置
################################################################################
scene_ids=(
    "20250818_134748_Q3707_200_230_part01",
    "20250818_134748_Q3707_600_630_part01",
    "20250818_134748_Q3707_90_120_part01",
    "20250818_140332_Q3707_1250_1280_part01",
    "20250818_140332_Q3707_990_1020_part01",
    "20250818_212913_Q3707_190_220_part01",
    "20250819_004633_Q3707_990_1010_part01",
    "20250819_015923_Q3707_100_130_part03",
    "20250819_091233_Q3708_2010_2040_part01",
    "20250819_190525_Q3708_130_160_part01",
    "20250822_164521_Q3714_1300_1330_part01",
    "20251025_163358_QCOYSD504206_1595_1610",
)

mode="${1:-0}"   # 0:线下模式, 1:线上模式
clip_time_start="${2:-0}"

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame, same in all scene_ids!

gpu=0
segformer_path=/opt/mmsegmentation
camera_ids=(0 1 2 3 5 6 7 9 10 11 12)
raw_data_root="/nas/scenario_data/" #_thoru
processed_data_root="/nas/scenario_output/zyk0525/processed"
work_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 训练配置
lidar_type="lidar" # lidar（运动补偿前）/visual（纯视觉）
output_root="output"
config_file="configs/omnire_extended_cam_lidar.yaml"
dataset_config="qcraft/11cams_${lidar_type}"
extra_config_info=""  # 额外信息，如 depth_loss
label_type="gt"  # pred使用自动化标注，gt使用人工标注

# 日志设置
PIPELINE_LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pipeline_log="${PIPELINE_LOG_DIR}/pipeline_results.txt"
echo "Pipeline 开始时间: $(date)" > "$pipeline_log"
echo "场景ID, 步骤, 状态, 时间" >> "$pipeline_log"
################################################################################

# step -1: data download, split and parse(only for online mode)
if [[ "$mode" == "1" ]]; then
    echo "[INFO] Online mode"
    echo "[INFO] clip_time_start: $clip_time_start"
    raw_data_root="/home/data/raw"
    processed_data_root="/home/data/processed"
    
    # 数据下载
    obs_root="obs://hwcn-hd2-ad-roaddata-new"
    AK="${OBS_AK:-}"
    SK="${OBS_SK:-}"
    ENDPOINT="obs.cn-east-4.myhuaweicloud.com"
    OBSUTIL="tools/obsutil_linux_amd64_5.8.3/obsutil"
    mkdir -p "${raw_data_root}"
    
    for scene_id in "${scene_ids[@]}"; do
        echo "[INFO] start downloading scene: ${scene_id}..."
        obs_dir="${obs_root}/${scene_id}"
        "${OBSUTIL}" cp "${obs_dir}" "${raw_data_root}/" -r -f -i="${AK}" -k="${SK}" -e="${ENDPOINT}"
        if [ $? -ne 0 ]; then
            echo "Error: Download failed for scene ${scene_id}"
            exit 1
        fi
        echo "[INFO] download success for scene: ${scene_id}"
    done

    # 数据切分和解析 TODO: 根据 clip_time_start 切分数据，并解析成后续处理需要的格式
else
    echo "[INFO] Offline mode"
fi

# step 0: sparse4d extract labels, TODO not check
if [ "${label_type}" = "pred" ]; then
    echo "Step 0: Extracting labels with Sparse4D..."
    source /opt/conda/etc/profile.d/conda.sh && conda activate sparse4d
    export LD_LIBRARY_PATH=/opt/conda/envs/sparse4d/lib:$LD_LIBRARY_PATH
    export TORCH_CUDA_ARCH_LIST="8.0"

    ln -s /opt/conda/envs/sparse4d/lib/python3.9/site-packages/deformable_aggregation_ext.cpython-39-x86_64-linux-gnu.so third_party/Sparse4D/projects/mmdet3d_plugin/ops
    abs_data_root=$(realpath "${raw_data_root}")

    for scene_id in "${scene_ids[@]}"; do
        echo "========================================="
        echo "Extracting labels for scene: ${scene_id}"
        echo "========================================="
        
        # 检查是否已存在，防止覆盖
        pred_label_dir="${abs_data_root}/${scene_id}/label_pred"
        if [ -d "$pred_label_dir" ]; then
            echo "警告：标签目录 $pred_label_dir 已存在，跳过此场景！"
            echo "$scene_id, step0_label, 跳过（已存在）, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
            continue
        fi
        
        cd third_party/Sparse4D
        CUDA_VISIBLE_DEVICES=${gpu} python infer_qcraft.py \
            --data_root ${abs_data_root} \
            --scene_idx ${scene_id}
        cd ../../
        
        if [ $? -ne 0 ]; then
            echo "Error: Label extraction failed for scene ${scene_id}"
            echo "$scene_id, step0_label, 失败, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
            exit 1
        fi
        echo "Step 0: OK"
        echo "$scene_id, step0_label, 成功, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
    done
else
    echo "Step 0: Skipping label extraction, using GT labels..."
fi

# step 1: preprocess
source /opt/conda/etc/profile.d/conda.sh && conda activate main
export PYTHONPATH=$(pwd)

for scene_id in "${scene_ids[@]}"; do
    echo "========================================="
    echo "Preprocessing scene: ${scene_id}"
    echo "========================================="
    
    # 检查是否已存在，防止覆盖
    images_dir="${processed_data_root}/training/${scene_id}/images"
    if [ -d "$images_dir" ]; then
        echo "警告：预处理结果 $images_dir 已存在，跳过此场景！"
        echo "$scene_id, step1_preprocess, 跳过（已存在）, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
        continue
    fi
    
    python datasets/preprocess.py \
        --data_root ${raw_data_root} \
        --target_dir ${processed_data_root} \
        --dataset qcraft \
        --split training \
        --scene_ids $scene_id \
        --workers 2 \
        --process_keys ego_masks images calib pose objects dynamic_masks lidar \
        --skip_front_wide_side_cameras \
        --label_type ${label_type}
    
    if [ $? -ne 0 ]; then
        echo "Error: Preprocessing failed for scene ${scene_id}"
        echo "$scene_id, step1_preprocess, 失败, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
        exit 1
    fi
    echo "Step 1: OK"
    echo "$scene_id, step1_preprocess, 成功, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
done

# step 2: extract mask
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh
source /opt/conda/etc/profile.d/conda.sh && conda activate mmseg2

# Pick an avaliable gpu
#source scripts/utils.sh
#if [ "${gpu}" = "-1" ]; then
#    gpu=$(pick_gpu)
#    if [ -z "${gpu}" ]; then
#        echo "no gpu found"
#        exit 1
#    fi
#fi
echo "Using GPU: ${gpu}"

for scene_id in "${scene_ids[@]}"; do
    echo "========================================="
    echo "Extracting masks for scene: ${scene_id}"
    echo "========================================="
    
    # 注意：--ignore_existing 已在 Python 脚本内部检查 fine_dynamic_masks 是否存在，存在则跳过
    CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks_mmseg2.py \
        --data_root ${processed_data_root}/training \
        --config=${segformer_path}/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py \
        --checkpoint=${segformer_path}/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth \
        --scene_ids $scene_id \
        --process_dynamic_mask \
        --ignore_existing
    
    if [ $? -ne 0 ]; then
        echo "Error: Mask extraction failed for scene ${scene_id}"
        echo "$scene_id, step2_mask, 失败, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
        exit 1
    fi
    echo "Step 2: OK(existed skip/succeed)"
    echo "$scene_id, step2_mask, 成功, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
done


# step 3: process human pose
source /opt/conda/etc/profile.d/conda.sh && conda activate 4d-humans
export PYTHONPATH=$PYTHONPATH:${work_root}/third_party/Humans4D

for scene_id in "${scene_ids[@]}"; do
    echo "========================================="
    echo "Processing human pose for scene: ${scene_id}"
    echo "========================================="
    
    # 检查是否已存在，防止覆盖
    smpl_pkl="${processed_data_root}/training/${scene_id}/humanpose/smpl.pkl"
    if [ -f "$smpl_pkl" ]; then
        echo "警告：人体姿态结果 $smpl_pkl 已存在，跳过此场景！"
        echo "$scene_id, step3_humanpose, 跳过（已存在）, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
        continue
    fi
    
    CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
        --data_root ${processed_data_root}/training \
        --dataset qcraft \
        --scene_id $scene_id
    
    if [ $? -ne 0 ]; then
        echo "Error: Human pose processing failed for scene ${scene_id}"
        echo "$scene_id, step3_humanpose, 失败, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
        exit 1
    fi
    echo "Step 3: OK"
    echo "$scene_id, step3_humanpose, 成功, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
done

# 4: train
source /opt/conda/etc/profile.d/conda.sh && conda activate main
################################################################################

for scene_id in "${scene_ids[@]}"; do
    echo "========================================="
    echo "Training for scene: ${scene_id}"
    echo "========================================="
    
    project_name="qcraft_${scene_id}"
    date_str=$(date +%Y%m%d)
    run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info:+_${extra_config_info}}"
    run_dir="${output_root}/${project_name}/${run_name}"
    
    # 检查输出目录是否存在，防止覆盖
    if [ -d "$run_dir" ]; then
        echo "错误：目录 $run_dir 已存在！"
        echo "$scene_id, step4_train, 跳过（已存在）, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
        continue  # 跳过当前场景，继续下一个
    fi
    
    mkdir -p "$run_dir"
    
    # 备份当前脚本
    script_name=$(basename "$0")
    backup_path="${run_dir}/${script_name}"
    cp "$0" "$backup_path"
    
    # 启动训练（保留你的 data.data_root）
    export PYTHONPATH=$(pwd)
    CUDA_VISIBLE_DEVICES=${gpu} python tools/train.py \
        --config_file $config_file \
        --output_root $output_root \
        --project $project_name \
        --run_name $run_name \
        dataset=$dataset_config \
        data.data_root=${processed_data_root}/training \
        data.scene_idx=$scene_id \
        data.start_timestep=$start_timestep \
        data.end_timestep=$end_timestep \
        data.pixel_source.cameras="[$(IFS=','; echo "${camera_ids[*]}")]"
    
    if [ $? -ne 0 ]; then
        echo "Error: Training failed for scene ${scene_id}"
        echo "$scene_id, step4_train, 失败, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
        continue  # 跳过当前场景，继续下一个
    fi
    echo "Step 4: OK"
    echo "$scene_id, step4_train, 成功, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
done


# step 5: LiDAR训练
echo "Step 5: Skip"
#echo "$scene_id, step5_lidar_train, 跳过, $(date '+%Y-%m-%d %H:%M:%S')" >> "$pipeline_log"
#source /opt/conda/etc/profile.d/conda.sh && conda activate main
#LIDAR_PROJECT_ROOT="${work_root}/lidargs"
#echo "LiDAR project root: ${LIDAR_PROJECT_ROOT}"
#export PYTHONPATH="${LIDAR_PROJECT_ROOT}:${PYTHONPATH}"
#
#echo "Data path: ${processed_data_root}/training/${scene_id}"
#
#lidar_cpt_path="${run_dir}/lidar_cpt"
## 检查输出目录是否存在，防止覆盖
#if [ -d "$lidar_cpt_path" ]; then
#    echo "错误：目录 $lidar_cpt_path 已存在！"
#    exit 1
#fi
#mkdir -p "$lidar_cpt_path"
#echo "LiDAR checkpoint path: ${lidar_cpt_path}"
#
#python ${LIDAR_PROJECT_ROOT}/train.py \
#    -s ${processed_data_root}/training/${scene_id} \
#    -m ${lidar_cpt_path} \
#    --caseid ${scene_id} \
#    --gpu ${gpu} \
#    --iterations ${train_lidar_iterations} \
#    --max_depth ${max_depth} \
#    --dataset chery \
#    --block_size ${block_size}

echo ""
echo "======================================================================"
echo "Pipeline 所有步骤完成！"
echo "======================================================================"
echo "结果汇总已保存到: $pipeline_log"
echo "======================================================================"
echo "Pipeline 结果汇总:"
cat "$pipeline_log"
echo "======================================================================"
