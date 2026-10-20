#!/bin/bash
################################################################################
# 数据下载、解析和切片
################################################################################
source scripts/qcraft/common.sh

scene_id="$1"
clip_start_time="${2:-0}"
clip_end_time="${3:-}"
mode="${4:-1}"

scene_id_s_e=${scene_id}_${clip_start_time}_${clip_end_time}

if [[ "$mode" == "1" ]]; then
    echo "[INFO] Online mode"
    
    mkdir -p ${parsed_data_root}
    mkdir -p ${parsed_data_root}/${scene_id}

    # 检查是否满足解析条件,不满足则下载数据
    need_download=false
    required_files=("camera_jpg_img.stf" "lidar_data.stf" "lite_msg.stf" "lite_run_info.pb.bin")
    for file in "${required_files[@]}"; do
        if [ ! -f "${raw_data_root}/${scene_id}/${file}" ]; then
            need_download=true
            snippets_flag=false
            break
        fi
    done

    if [ "$need_download" = false ]; then
        echo "[INFO] All required files exist, skipping download"
    else
        # 数据下载
        obs_root="obs://hwcn-hd2-ad-roaddata-new"
        obs_dir="${obs_root}/${scene_id}"
        local_dir="${raw_data_root}/"
        mkdir -p "${raw_data_root}"

        # 重试机制：最多重试3次
        max_retries=3
        retry_count=0
        download_success=false
        
        while [ $retry_count -lt $max_retries ] && [ "$download_success" = false ]; do
            if [ $retry_count -gt 0 ]; then
                echo "[INFO] Retry download (attempt $((retry_count+1))/$max_retries)..."
                # 等待时间逐渐增加：10s, 20s, 30s
                wait_time=$((retry_count * 10))
                echo "[INFO] Waiting ${wait_time} seconds before retry..."
                sleep $wait_time
            fi
            
            echo "[INFO] start downloading (attempt $((retry_count+1))/$max_retries)..."
            # 在 obs_dir 里面找一下 required_files 里面的东西是不是都存在
            missing_files=()
            for file in "${required_files[@]}"; do
                full_path="${obs_dir}/${file}"
                if "${OBSUTIL}" stat "$full_path" -i="${AK}" -k="${SK}" -e="${ENDPOINT}" >/dev/null 2>&1; then
                    continue
                else 
                    missing_files+=("${file}")
                fi
            done

            # 如果permission denied : 使用 sudo chmod 777 -R tools
            if [ ${#missing_files[@]} -eq 0 ]; then
                # 都存在则走常规下载和流程
                snippets_flag=false
                "${OBSUTIL}" cp "${obs_dir}" "${local_dir}" -r -f -i="${AK}" -k="${SK}" -e="${ENDPOINT}" 2>&1 | grep -v -E "$OBS_FILTER" || true
            else
                # 有一个不存在就去snippets里面找
                echo "[INFO] Some files missing, searching in snippets..."
                snippets_flag=true
                python3 ./third_party/StfTools/download_snippets.py \
                    --scene_id "${scene_id}" \
                    --clip_start_time "${clip_start_time}" \
                    --clip_end_time "${clip_end_time}" \
                    --output_dir "${raw_data_root}/${scene_id}" \
                    --obsutil "${OBSUTIL}" \
                    --ak "${AK}" --sk "${SK}" --endpoint "${ENDPOINT}"
            fi

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "[INFO] download success"
                download_success=true
            else
                echo "[WARNING] download failed (attempt $((retry_count+1))/$max_retries)"
                retry_count=$((retry_count + 1))
                
                # 如果是最后一次尝试失败，则报错退出
                if [ $retry_count -eq $max_retries ]; then
                    echo "[ERROR] All $max_retries download attempts failed"
                    exit 1
                fi
            fi
        done
    fi
    
    snippets_dir="${raw_data_root}/${scene_id}/${scene_id_s_e}"
    #-1.1 数据切片(原始打包需要,snippets不需要)
    if [ "$snippets_flag" = true ]; then
       ########## snippets 为 True ,运行merging and cutting 脚本
        echo "[INFO] Merge and cut Snippets"
        python ./third_party/StfTools/snippets_merging.py \
            "--download_dir" "${snippets_dir}/download" \
            "--clip_start_time" "${clip_start_time}" \
            "--clip_end_time" "${clip_end_time}" \
            "--output" "${snippets_dir}"

        stf_files=("camera_jpg_img.stf" "lidar_data.stf" "lite_msg.stf")
        run_dir="${raw_data_root}/${scene_id}"

    else
        stf_files=("camera_jpg_img.stf" "lidar_data.stf" "lite_msg.stf")
        run_dir="${raw_data_root}/${scene_id}"
        for file in "${stf_files[@]}"; do
            echo "Split Processing: $file"
            python ./third_party/StfTools/stf_split_stream.py \
                "${run_dir}/${file}" \
                --time_range ${clip_start_time} ${clip_end_time} \
                --output "${snippets_dir}"
            
            if [ $? -ne 0 ]; then
                echo "Error processing $file"
                exit 1
            fi
        done
        cp "${run_dir}/lite_run_info.pb.bin" "${snippets_dir}/lite_run_info.pb.bin"
    fi

    # -1.2 数据解析
    
    # 解析图片
    echo "[INFO] Parsing Images Time:${clip_start_time} - ${clip_end_time}"
    python ./third_party/StfTools/stf_extract_h265_to_jpg.py \
        "${snippets_dir}/camera_jpg_img_${clip_start_time}_${clip_end_time}.stf" \
        --output "${parsed_data_root}/${scene_id}/${scene_id_s_e}"

    # 解析点云
    echo "[INFO] Parsing PointCloud Time:${clip_start_time} - ${clip_end_time}"
    python ./third_party/StfTools/stf_extract_pointcloud.py \
        "${snippets_dir}" \
        "lidar_data_${clip_start_time}_${clip_end_time}.stf" \
        --output ${parsed_data_root}/${scene_id}/pcd_${clip_start_time}_${clip_end_time} \
        --apply-extrinsics
    # 解析位姿
    echo "[INFO] Parsing LiteMsg Time:${clip_start_time} - ${clip_end_time}"
    python ./third_party/StfTools/stf_to_json_full.py \
        "${snippets_dir}/lite_msg_${clip_start_time}_${clip_end_time}.stf" \
        --output "${parsed_data_root}/${scene_id}"
    # 解析传感器
    echo "[INFO] Parsing Sensors Parameters"
    python ./third_party/StfTools/extract_sensor_params.py \
        "${snippets_dir}" \
        --output "${parsed_data_root}/${scene_id}/${scene_id_s_e}"
    # img与pcd文件组织
    python ./third_party/StfTools/img_pcd_organize.py \
        --parsed_data_root  ${parsed_data_root} \
        --scene_id ${scene_id} \
        --time_range ${clip_start_time} ${clip_end_time}
    # 提取3dbox标注
    python ./third_party/StfTools/3dbox_extract.py \
        --parsed_data_root  ${parsed_data_root} \
        --scene_id ${scene_id} \
        --time_range ${clip_start_time} ${clip_end_time}
    
    rm -rf ${parsed_data_root}/${scene_id}/pcd_${clip_start_time}_${clip_end_time}
    rm -rf ${parsed_data_root}/${scene_id}/lite_msg_${clip_start_time}_${clip_end_time}

    # 手动添加ego_mask
    cp -r /nas/scenario_output/zhangyingjun/data/ego_mask/20260616_155844_QCJPTDJ09478/ego_masks ${parsed_data_root}/${scene_id}/${scene_id_s_e}/
fi

if [[ "$mode" == "0" ]]; then
    echo "[INFO] Offline mode: Using local data"
    echo "[INFO] scene_id: $scene_id"
fi