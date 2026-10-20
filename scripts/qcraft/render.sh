scene_id="$1"
mode="${2:-1}"  # 0:线下模式, 1:线上模式
version="$3"    # 代码版本
################################################################################
camera_ids=(0 1 2 3 5 6 7 9 10 11 12)
lidar_type="lidar"
extra_config_info=""

project_name="qcraft_${scene_id}"

data_root="/home/data"
OBS_FILTER='\[[=>_ \-]+\]|[0-9]+\.[0-9]+%|tps:|Succeed count|Failed count|Succeed bytes|Metrics|Task id| [0-9]+/[0-9]+ |[0-9]+\.[0-9]+KB/s|[0-9]+\.[0-9]+MB/[0-9]+\.[0-9]+MB'

# 避免没有权限
chmod -R 755 tools/obsutil_linux_amd64_5.8.3

unset http_proxy
unset https_proxy

# 定义带重试的下载函数
download_with_retry() {
    local obs_path="$1"
    local local_path="$2"
    local description="$3"
    local max_retries=3
    local retry_count=0
    local success=0
    
    echo "[INFO] Downloading ${description}..."
    echo "[INFO] From: ${obs_path}"
    echo "[INFO] To: ${local_path}"
    
    while [ $retry_count -lt $max_retries ] && [ $success -eq 0 ]; do
        if [ $retry_count -gt 0 ]; then
            echo "[WARN] Retry attempt $retry_count/$max_retries for ${description}..."
            sleep $((retry_count * 5))
        fi
        
        mkdir -p "$(dirname ${local_path})" 2>/dev/null || true
        
        ${OBSUTIL} cp "${obs_path}" "${local_path}" -r -f \
            -i="${AK}" -k="${SK}" -e="${ENDPOINT}" 2>&1 | grep -vE "${OBS_FILTER}"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "[INFO] ${description} download success on attempt $((retry_count+1))"
            success=1
        else
            echo "[ERROR] ${description} download failed on attempt $((retry_count+1))"
            retry_count=$((retry_count + 1))
            rm -rf "${local_path}" 2>/dev/null || true
        fi
    done
    
    if [ $success -eq 0 ]; then
        echo "[ERROR] ${description} download failed after $max_retries attempts"
        return 1
    fi
    return 0
}

# 定义获取最新的 run_name 函数
get_latest_run_from_obs() {
    local obs_output_path="$1"

    "${OBSUTIL}" ls "${obs_output_path}" \
        -i="${AK}" \
        -k="${SK}" \
        -e="${ENDPOINT}" \
        -d \
    | grep -oE '[0-9]{8}_lidar\+(train)?cam[0-9_]+' \
    | LC_ALL=C sort -ru \
    | head -n 1
}

if [[ "$mode" == "1" ]]; then
    echo "[INFO] Online mode"

    processed_data_root="${data_root}/processed"
    training_output_root="${data_root}/output"

    # OBS 配置
    OBS_BUCKET="obs://hwcn-hd2-ad-simulation"
    OBS_BASE_PATH="scene_reconstruction"
    OBSUTIL="tools/obsutil_linux_amd64_5.8.3/obsutil"
    AK="${OBS_AK:-}"
    SK="${OBS_SK:-}"
    ENDPOINT="obs.cn-east-4.myhuaweicloud.com"

    # 下载1: 预处理数据
    download_with_retry \
        "${OBS_BUCKET}/${OBS_BASE_PATH}/${scene_id}/${version}/preprocess/${scene_id}/" \
        "${processed_data_root}/" \
        "preprocessed data"
    if [ $? -ne 0 ]; then
        exit 1
    fi

    # 下载2: data_frame_car_info.json
    download_with_retry \
        "${OBS_BUCKET}/${OBS_BASE_PATH}/${scene_id}/${version}/parsed/data_frame_car_info.json" \
        "${processed_data_root}/${scene_id}/" \
        "data_frame_car_info.json"
    if [ $? -ne 0 ]; then
        exit 1
    fi

    # 从 OBS 获取最新的 run_name
    obs_output_path="${OBS_BUCKET}/${OBS_BASE_PATH}/${scene_id}/${version}/output/"
    run_name=$(get_latest_run_from_obs "${obs_output_path}")
    if [ $? -ne 0 ] || [ -z "${run_name}" ]; then
        exit 1
    fi
    echo "[INFO] Found latest run_name from OBS: ${run_name}"

    # 下载3: 训练输出
    download_with_retry "${OBS_BUCKET}/${OBS_BASE_PATH}/${scene_id}/${version}/output/${run_name}/group0/checkpoint_final.pth" "${training_output_root}/${project_name}/${run_name}/group0/" "group0 checkpoint_final.pth"
    download_with_retry "${OBS_BUCKET}/${OBS_BASE_PATH}/${scene_id}/${version}/output/${run_name}/group1/checkpoint_final.pth" "${training_output_root}/${project_name}/${run_name}/group1/" "group1 checkpoint_final.pth"
    download_with_retry "${OBS_BUCKET}/${OBS_BASE_PATH}/${scene_id}/${version}/output/${run_name}/lidar_cpt/" "${training_output_root}/${project_name}/${run_name}/" "lidar_cpt"
    download_with_retry "${OBS_BUCKET}/${OBS_BASE_PATH}/${scene_id}/${version}/output/${run_name}/group0/config.yaml" "${training_output_root}/${project_name}/${run_name}/group0/" "group0 config.yaml"
    download_with_retry "${OBS_BUCKET}/${OBS_BASE_PATH}/${scene_id}/${version}/output/${run_name}/group1/config.yaml" "${training_output_root}/${project_name}/${run_name}/group1/" "group1 config.yaml"

    # 修改 config.yaml 中的 data_root
    CONFIG_FILE_GROUP0="${training_output_root}/${project_name}/${run_name}/group0/config.yaml"
    CONFIG_FILE_GROUP1="${training_output_root}/${project_name}/${run_name}/group1/config.yaml"
    NEW_DATA_ROOT="${processed_data_root}"

    if [ -f "$CONFIG_FILE_GROUP0" ]; then
        echo "[INFO] Updating group0 config.yaml data_root..."

        sed -i "s|^\([[:space:]]*\)data_root:.*|\1data_root: ${NEW_DATA_ROOT}|g" ${CONFIG_FILE_GROUP0}
        echo "[INFO] Updated group0 data_root to: ${NEW_DATA_ROOT}"
    fi

    if [ -f "$CONFIG_FILE_GROUP1" ]; then
        echo "[INFO] Updating group1 config.yaml data_root..."

        sed -i "s|^\([[:space:]]*\)data_root:.*|\1data_root: ${NEW_DATA_ROOT}|g" ${CONFIG_FILE_GROUP1}
        echo "[INFO] Updated group1 data_root to: ${NEW_DATA_ROOT}"
    fi

    checkpoint_path_group0="${training_output_root}/${project_name}/${run_name}/group0/checkpoint_final.pth"
    checkpoint_path_group1="${training_output_root}/${project_name}/${run_name}/group1/checkpoint_final.pth"
    config_path_group0="${training_output_root}/${project_name}/${run_name}/group0/config.yaml"
    config_path_group1="${training_output_root}/${project_name}/${run_name}/group1/config.yaml"
    lidar_checkpoint_path="${training_output_root}/${project_name}/${run_name}/lidar_cpt"
fi

if [[ "$mode" == "0" ]]; then
    echo "[INFO] Offline mode"

    processed_data_root="${data_root}/processed"
    checkpoint_path_group0="${data_root}/output/${project_name}/${run_name}/group0/checkpoint_final.pth"
    checkpoint_path_group1="${data_root}/output/${project_name}/${run_name}/group1/checkpoint_final.pth"
    config_path_group0="${data_root}/output/${project_name}/${run_name}/group0/config.yaml"
    config_path_group1="${data_root}/output/${project_name}/${run_name}/group1/config.yaml"
    lidar_checkpoint_path="${data_root}/output/${project_name}/${run_name}/lidar_cpt"
fi

# step 0: 环境初始化
sleep 20
cd /home/scene_reconstruction
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd lidargs/
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..
# conda init
# sleep 10
source /opt/conda/etc/profile.d/conda.sh
conda activate main

# step 1: 启动渲染服务器
echo "processed_data_root: $processed_data_root"

cd /opt/LiDAR-GS/submodules/
pip uninstall diff_lidargs_surfel_rasterization -y
# 确认输出正常
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
#清理
rm -rf ./diff_lidargs_surfel_rasterization/build
rm -rf ~/.cache/torch_extensions/
rm -rf /tmp/pip-build-env-*
#重新编译
pip install ./diff_lidargs_surfel_rasterization --no-build-isolation
cd /home/scene_reconstruction
python sim_render/server.py \
    --resume_from_group0 ${checkpoint_path_group0} \
    --resume_from_group1 ${checkpoint_path_group1} \
    --config_file_group0 ${config_path_group0} \
    --config_file_group1 ${config_path_group1} \
    --source_path ${processed_data_root}/${scene_id} \
    --lidar_checkpoint_path ${lidar_checkpoint_path} \
    --output_dir render_output
echo "[INFO] Render completed"
