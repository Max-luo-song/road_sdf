#!/bin/bash
################################################################################
# 上传结果到OBS
################################################################################
source scripts/qcraft/common.sh

scene_id="$1"
clip_start_time="${2:-0}"
clip_end_time="${3:-}"
version="${4:-26.5.20.1}"
run_dir="$5"

scene_id_s_e=${scene_id}_${clip_start_time}_${clip_end_time}

echo "[INFO] Uploading to OBS..."

"${OBSUTIL}" cp \
    "${processed_data_root}/training/${scene_id_s_e}" \
    "obs://hwcn-hd2-ad-simulation/scene_reconstruction/${scene_id_s_e}/${version}/preprocess/" \
    -r -f \
    -i="${AK}" \
    -k="${SK}" \
    -e="${ENDPOINT}" 2>&1 | grep -v -E "$OBS_FILTER" || true

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[INFO] Processed data uploaded successfully"
else
    echo "[ERROR] Processed data upload failed"
    exit 1
fi

echo "[INFO] Uploading parsed data..."
parsed_json_path="${parsed_data_root}/${scene_id}/${scene_id_s_e}/data_frame_car_info.json"
"${OBSUTIL}" cp \
    "${parsed_json_path}" \
    "obs://hwcn-hd2-ad-simulation/scene_reconstruction/${scene_id_s_e}/${version}/parsed/data_frame_car_info.json" \
    -f \
    -i="${AK}" \
    -k="${SK}" \
    -e="${ENDPOINT}" 2>&1 | grep -v -E "$OBS_FILTER" || true

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[INFO] Parsed data uploaded successfully"
else
    echo "[ERROR] Parsed data upload failed"
    exit 1
fi

echo "[INFO] Uploading output data..."
"${OBSUTIL}" cp \
    "${run_dir}" \
    "obs://hwcn-hd2-ad-simulation/scene_reconstruction/${scene_id_s_e}/${version}/output/" \
    -r -f \
    -i="${AK}" \
    -k="${SK}" \
    -e="${ENDPOINT}" 2>&1 | grep -v -E "$OBS_FILTER" || true

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[INFO] Output data uploaded successfully"
else
    echo "[ERROR] Output data upload failed"
    exit 1
fi

echo "[INFO] Upload completed"