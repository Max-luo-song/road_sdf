iterations=1000
gpu=0
# 预处理之后的文件夹
data="/home/not0513/data/qcraft/processed/training/20251025_163358_QCOYSD504206_1595_1610"
caseid="20251025_163358_QCOYSD504206_1595_1610"
# 模型文件路径，参考data目录，training改为test，再按照caseid/日期+时间自动生成output_dir
test_str="/home/not0513/data/qcraft/processed/test"
date_str="2026-01-06-11-06"
output_dir="${test_str}/${caseid}/${date_str}"
# 场景编辑yaml文件路径
edit_yaml="/scene_reconstruction/edit_config.yaml"
block_size=50
# 渲染输出路径
output_path="${output_dir}/render"

# 将项目根目录加入 PYTHONPATH(上两级)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python3 $PROJECT_ROOT/sim_render/lidar/render.py -s ${data} -m ${output_dir} --caseid ${caseid} \
                  --iteration ${iterations} --max_depth 100 \
                  --edit_yaml ${edit_yaml} --block_size ${block_size} \
                  --output_path ${output_path}