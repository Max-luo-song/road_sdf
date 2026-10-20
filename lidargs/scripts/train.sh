#!/bin/bash
trap 'exit 1' INT

scene_ids=(
    "20251121_095534_QCOYPDA27344_1538_1553"
)

iterations=5000
gpu=0
mode="120"    # 120: HFOV=120° (dataset=chery)  |  360: HFOV=360° (dataset=chery_lidar_360)

PREPROCESS_TARGET="/nas_thoru/scenario_output/yangtao/data/qcraft/processed"
test_str="/nas_thoru/oldbak/yx/lidar630/cheryoutput/120"
output_suffix="2026-512"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ "${mode}" = "360" ]; then
    dataset="chery_lidar_360"   # 360° 全景, hfov=2π
elif [ "${mode}" = "120" ]; then
    dataset="chery"             # 120° 前向, hfov=2π/3
else
    echo "[ERROR] mode must be '120' or '360', got: ${mode}"; exit 1
fi

enable_raydrop_unet=false
if ${enable_raydrop_unet}; then
    ENABLE_RAYDROP_UNET="--enable_raydrop_unet"
    echo "Enable Raydrop UNet."
else
    ENABLE_RAYDROP_UNET=""
    echo "Raydrop UNet disabled."
fi

for scene_id in "${scene_ids[@]}"; do
    data="${PREPROCESS_TARGET}/training/${scene_id}"
    caseid="${scene_id}"
    output_dir="${test_str}/${caseid}/${output_suffix}"

    echo "========================================"
    echo "Training scene: ${scene_id}"
    echo "Data path: ${data}"
    echo "Output path: ${output_dir}"
    echo "========================================"

    mkdir -p "${output_dir}"
    LOG_FILE="${output_dir}/console.log"

    python3 "${PROJECT_ROOT}"/lidargs/train.py \
        -s "${data}" \
        -m "${output_dir}" \
        --caseid "${caseid}" \
        --gpu "${gpu}" \
        --iterations "${iterations}" \
        --max_depth 120 \
        --dataset ${dataset} \
        --block_size 50 \
        ${ENABLE_RAYDROP_UNET} \
        2>&1 | tee -a "${LOG_FILE}"

    if ${enable_raydrop_unet}; then
        python3 "${PROJECT_ROOT}"/lidargs/refine_ray_drop.py \
            -s "${data}" \
            -m "${output_dir}" \
            --gpu "${gpu}"
    fi

    echo "========================================"
    echo "Training completed for scene: ${scene_id}"
    echo "========================================"

done