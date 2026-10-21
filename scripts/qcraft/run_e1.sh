#!/bin/bash
# e1: one learned-height run, group0 only, followed by original / left-3m renders.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
scene="${1:-20260613_102138_QCJPTDJ07384_3298_3308}"
export GROUP0_GPU="${E1_GPU:-0}"
export QCRAFT_CONFIG_FILE="configs/road_e1.yaml"
export RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_e1_sdf-road-gs-fixedz}"
if [[ ! "${scene}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ || ! "${RUN_NAME_OVERRIDE}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    echo "[ERROR] Invalid scene or run name." >&2
    exit 1
fi
if ! command -v python >/dev/null && [[ -x /root/miniconda3/bin/python ]]; then
    export PATH="/root/miniconda3/bin:${PATH}"
fi
source scripts/qcraft/common.sh
if [[ -n "${RESUME_FROM_GROUP0:-}" ]]; then
    if [[ ! -f "${RESUME_FROM_GROUP0}" ]]; then
        echo "[ERROR] Resume checkpoint not found: ${RESUME_FROM_GROUP0}" >&2
        exit 1
    fi
    printf 'Resume group0 from: %s\n' "${RESUME_FROM_GROUP0}"
fi
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
python - <<'PYCONFIG'
from utils.config import load_config
cfg = load_config("configs/road_e1.yaml")
field = cfg.model.RoadNodes.networks.road_surface
assert field.enabled and field.learn_height and field.align_normals and field.free_scales and field.free_opacity and field.fixed_normal_scale == 0.005, "e1 requires learned SDF normals, free XY scales/opacity, and fixed normal scale"
assert cfg.trainer.losses.road_color.w > 0 and cfg.trainer.losses.road_coverage.w > 0 and cfg.trainer.losses.road_contribution.w > 0
print(f"e1: learned SDF + road GS appearance; {cfg.trainer.optim.num_iters} steps; group0 cameras 2,3,5,9,12")
PYCONFIG
expected_run_dir="${output_root}/qcraft_${scene}/${RUN_NAME_OVERRIDE}"
if [[ -e "${expected_run_dir}" ]]; then
    echo "[ERROR] Run already exists: ${expected_run_dir}. Set a new RUN_NAME_OVERRIDE." >&2
    exit 1
fi
if [[ ! -d "${processed_data_root}/training/${scene}" ]]; then
    echo "[ERROR] Processed scene not found: ${processed_data_root}/training/${scene}" >&2
    exit 1
fi
printf 'Output: %s\n' "${expected_run_dir}"
if [[ "${E1_DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY RUN] Would train/evaluate group0, then render original_traj and left_shift_3m."
    exit 0
fi
run_dir="$(bash scripts/qcraft/group_training.sh "${scene}" group0)"
env run_dir="${run_dir}" RENDER_GPU="${GROUP0_GPU}" \
    traj_types_csv=original_traj,left_shift_3m \
    bash sim_render/cam/render_novel_trajectory.sh
printf 'e1 complete: %s\n' "${run_dir}"
