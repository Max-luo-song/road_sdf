#!/bin/bash
trap 'exit 1' INT
# renderedit.sh — 批量 LiDAR 渲染（默认 120 前向，纯渲染）
# 用法: bash lidargs/scripts/renderedit.sh

# 【必填】Clip 列表 & 对应 YAML 列表（一一对应，数量必须相同）

CLIPS=(
    "/nas/oldbak/yx/lidar630/cheryoutputcc/120/20250818_140332_Q3707_1250_1280_part01"
)

YAMLS=(
    "/nas/oldbak/yx/M5finalfix/scene_reconstruction/edit_config.yaml"   # 仅 enable_edit=1 时使用
)

# 源数据路径来源：0=自动从模型 cfg_args 回填（默认；预处理数据按训练时的绝对路径就在本机时用）
#               1=手动指定下方 SOURCE_PATHS
# 何时用 1：不同服务器挂载盘命名不同（如本机只挂 /nas、而 cfg 里写死的是 /nas_thoru），
#          导致 cfg 里的 source_path 在本机不存在、渲染报 .../lidar/bin 找不到时，改用手填。
manual_source=0
# manual_source=1 时必填：与 CLIPS 一一对应的源数据路径（预处理输出目录，内含 lidar/bin）
SOURCE_PATHS=(
    "/nas/oldbak/yx/cherydatapreprocess/training/20250818_140332_Q3707_1250_1280_part01"
)

# 【必填】集中日志目录
LOG_DIR="/nas/oldbak/yx/lidar630/cheryoutput/120/batch_logs"

# 公共参数
block_size=50
iteration=-1
max_depth=100

enable_edit=0    # 1=启用编辑  0=纯渲染
no_video=0       # 1=跳过 BEV 视频生成
no_txt=1         # 1=跳过 TXT 点云写入（np.savetxt），大幅加速（建议与 fastmode 一起开启）
mode="120"    # 120: HFOV=120° → dataset=chery  |  360: HFOV=360° → dataset=chery_lidar_360

# 快速渲染：1=传 --fast_render 启用 renderedit.py 内部 GPU pano2lidar 加速（建议与 no_txt 一起开）
# v0.1.9 的 stream-safe 多流已在 renderer 内默认启用，无需再前插/选预编译扩展
fastmode=1

# BEV 视频参数
video_fps=10
bev_width=1200
bev_height=900
bev_ymin=-30
bev_ymax=30
bev_point_size=0.2
# 120: x 轴 (-10, 50)，自车在图像下方约 17%，前方视野更宽
# 360: 不传 xmin/xmax，用默认值
if [ "${mode}" = "120" ]; then
    bev_xmin=-10
    bev_xmax=50
else
    bev_xmin=""
    bev_xmax=""
fi

# 后处理
enable_raydrop_postfilter=0
raydrop_postfilter_threshold=0.4
enable_depth_postfilter=1
depth_postfilter_ratio_threshold=0.10
near_preserve_depth=0.0

# 最大并行 job 数（1=串行）
MAX_PARALLEL=1

# ------------------------------------------------------------------------------

if [ "${#CLIPS[@]}" -ne "${#YAMLS[@]}" ]; then
    echo "[ERROR] CLIPS 和 YAMLS 数量不一致 (${#CLIPS[@]} vs ${#YAMLS[@]})，退出。"
    exit 1
fi

if [ "${manual_source}" = "1" ] && [ "${#CLIPS[@]}" -ne "${#SOURCE_PATHS[@]}" ]; then
    echo "[ERROR] manual_source=1 但 SOURCE_PATHS 与 CLIPS 数量不一致 (${#SOURCE_PATHS[@]} vs ${#CLIPS[@]})，退出。"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ "${mode}" = "360" ]; then
    dataset="chery_lidar_360"   # 360° 全景, hfov=2π
elif [ "${mode}" = "120" ]; then
    dataset="chery"             # 120° 前向, hfov=2π/3
else
    echo "[ERROR] mode must be '120' or '360', got: ${mode}"; exit 1
fi

mkdir -p "${LOG_DIR}"

TOTAL="${#CLIPS[@]}"
BATCH_START="$(date '+%Y-%m-%d %H:%M:%S')"
MASTER_LOG="${LOG_DIR}/batch_master.log"

{
echo "============================================================"
echo "  LiDAR Batch Edit Renderer  —  共 ${TOTAL} 个 clip"
echo "  并行数  : ${MAX_PARALLEL}"
echo "  开始时间: ${BATCH_START}"
echo "  日志目录: ${LOG_DIR}"
echo "============================================================"
} | tee "${MASTER_LOG}"

run_one() {
    local idx="$1"
    local model_path="${CLIPS[$idx]}"
    local edit_config="${YAMLS[$idx]}"
    local output_path="${model_path}/render_edit"

    local scene_name
    scene_name="$(basename "$(dirname "${model_path}")")_$(basename "${model_path}")"
    local log_file="${LOG_DIR}/render_${scene_name}.log"

    mkdir -p "${output_path}"

    {
    echo ""
    echo "------------------------------------------------------------"
    echo "  [$(( idx + 1 ))/${TOTAL}] 开始: $(date '+%H:%M:%S')"
    echo "  model_path : ${model_path}"
    echo "  edit_config: ${edit_config}"
    echo "  output_path: ${output_path}"
    echo "  log        : ${log_file}"
    echo "------------------------------------------------------------"
    } | tee -a "${MASTER_LOG}"

    if [ ! -d "${model_path}" ]; then
        echo "  [SKIP] model_path 不存在: ${model_path}" | tee -a "${MASTER_LOG}"
        return
    fi

    if [ "${enable_edit}" = "1" ] && [ ! -f "${edit_config}" ]; then
        echo "  [WARN] edit_config 不存在: ${edit_config}，跳过本 clip" | tee -a "${MASTER_LOG}"
        return
    fi

    CMD="python3 ${PROJECT_ROOT}/lidargs/renderedit.py \
        -m \"${model_path}\" \
        --output_path \"${output_path}\" \
        --dataset ${dataset} \
        --block_size ${block_size} \
        --iteration ${iteration} \
        --max_depth ${max_depth} \
        --video_fps ${video_fps} \
        --bev_width ${bev_width} \
        --bev_height ${bev_height} \
        --bev_ymin ${bev_ymin} \
        --bev_ymax ${bev_ymax} \
        --bev_point_size ${bev_point_size} \
        --raydrop_postfilter_threshold ${raydrop_postfilter_threshold} \
        --depth_postfilter_ratio_threshold ${depth_postfilter_ratio_threshold} \
        --near_preserve_depth ${near_preserve_depth}"

    [ "${manual_source}" = "1" ]             && CMD="${CMD} -s \"${SOURCE_PATHS[$idx]}\""
    [ "${enable_edit}" = "1" ]               && CMD="${CMD} --edit_config \"${edit_config}\""
    [ -n "${bev_xmin}" ]                     && CMD="${CMD} --bev_xmin ${bev_xmin} --bev_xmax ${bev_xmax}"
    [ "${enable_raydrop_postfilter}" = "1" ] && CMD="${CMD} --enable_raydrop_postfilter"
    [ "${enable_depth_postfilter}"   = "1" ] && CMD="${CMD} --enable_depth_postfilter"
    [ "${no_video}"                  = "1" ] && CMD="${CMD} --no_video"
    [ "${no_txt}"                    = "1" ] && CMD="${CMD} --no_txt"
    [ "${fastmode}"                  = "1" ] && CMD="${CMD} --fast_render"

    eval ${CMD} 2>&1 | tee "${log_file}"
    local exit_code="${PIPESTATUS[0]}"

    if [ "${exit_code}" -eq 0 ]; then
        echo "  [OK ] clip $(( idx + 1 )) 完成: $(date '+%H:%M:%S')  → ${log_file}" | tee -a "${MASTER_LOG}"
    else
        echo "  [ERR] clip $(( idx + 1 )) 失败 (exit=${exit_code}): $(date '+%H:%M:%S')  → ${log_file}" | tee -a "${MASTER_LOG}"
    fi
}

job_count=0
for i in "${!CLIPS[@]}"; do
    if [ "${MAX_PARALLEL}" -gt 1 ]; then
        run_one "$i" &
        job_count=$(( job_count + 1 ))
        if [ "${job_count}" -ge "${MAX_PARALLEL}" ]; then
            wait
            job_count=0
        fi
    else
        run_one "$i"
    fi
done

wait

{
echo ""
echo "============================================================"
echo "  全部 ${TOTAL} 个 clip 处理完毕: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志汇总: ${MASTER_LOG}"
echo "  各 clip 详细日志: ${LOG_DIR}/render_*.log"
echo "============================================================"
} | tee -a "${MASTER_LOG}"
