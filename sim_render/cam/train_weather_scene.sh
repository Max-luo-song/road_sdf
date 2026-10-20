# 参数设置
################################################################################
gpu=-1

weather_type="snowy" # only support "snowy", "rainy", "foggy".

scene_idx="20250818_134748_Q3707_200_230_part03"

camera_ids=(2 5 9)  # camera IDs to use, e.g., (0), (0 2 4)
# camera_ids=(0)  # camera IDs to use, e.g., (0), (0 2 4)
lidar_type="lidar"  # lidar（运动补偿前）/visual（纯视觉）

config_file="configs/omnire_extended_cam_lidar.yaml"
dataset_config="qcraft/3cams_${lidar_type}"
extra_config_info="depth_loss"  # 额外信息，如 depth_loss

start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
################################################################################

# Pick an avaliable gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
    fi
fi
echo "Using GPU: ${gpu}"


output_root="/nas_thoru/users/liuzy/data/scene_reconstruction_data/10clips/output"
project_name="qcraft_${scene_idx}"

date_str=$(date +%Y%m%d)
run_name="${date_str}_${lidar_type}+cam$(IFS="_"; echo "${camera_ids[*]}")${extra_config_info}_${weather_type}"

run_dir="${output_root}/${project_name}/${run_name}"

# 检查输出目录是否存在，防止覆盖
if [ -d "$run_dir" ]; then
    echo "错误：目录 $run_dir 已存在！"
    rm -r "$run_dir"
    # exit 1
fi
mkdir -p "$run_dir"

# 备份当前脚本到输出目录
script_name=$(basename "$0")
backup_path="${run_dir}/${script_name}"
cp "$0" "$backup_path"

# 启动训练
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=${gpu} python sim_render/cam/train_weather_scene.py \
    --config_file $config_file \
    --output_root $output_root \
    --project $project_name \
    --run_name $run_name \
    --weather_type $weather_type \
    dataset=$dataset_config \
    data.scene_idx=$scene_idx \
    data.pixel_source.cameras="[$(IFS=','; echo "${camera_ids[*]}")]" \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep