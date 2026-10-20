# 参数设置
################################################################################
SCENE_IDX="20251105_152839_QCOYSD504206_1240_1255"

DATA_DIR="data/qcraft/processed/training/$SCENE_IDX"
OUTPUT_DIR="output/qcraft_$SCENE_IDX"
################################################################################

export PYTHONPATH=$(pwd)
python datasets/qcraft/qcraft_visualize_masks.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --use_fine_dynamic_mask