# 参数设置
################################################################################
weather_type="snowy"
output_dir="/nas_thoru/users/liuzy/data/scene_reconstruction_data/10clips/processed/training/20250818_134748_Q3707_200_230_part03/images_$weather_type"
data_root="/nas_thoru/users/liuzy/data/scene_reconstruction_data/10clips/processed/training/20250818_134748_Q3707_200_230_part03/images"
cams=(2 5 9)
################################################################################

# qcraft dataset
export PYTHONPATH=$(pwd)
python WeatherEdit/background_editing/src/inference.py \
        --output_dir $output_dir \
        --dataset qcraft \
        --weather_type $weather_type  \
        --data_root $data_root \
        --cams ${cams[@]} \