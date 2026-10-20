#!/bin/bash
# DiFix progressive schedule 构建工具。
#
# shift token 规范（逗号分隔）：
#   left_1  left_2  left_3   -> left_shift_1m / 2m / 3m
#   right_1 right_2 right_3  -> right_shift_1m / 2m / 3m
#
# pipeline 默认 6 条：left_1,left_2,left_3,right_1,right_2,right_3

_difix_shift_token_to_traj_type() {
    case "$1" in
        left_1)  echo "left_shift_1m"  ;;
        left_2)  echo "left_shift_2m"  ;;
        left_3)  echo "left_shift_3m"  ;;
        right_1) echo "right_shift_1m" ;;
        right_2) echo "right_shift_2m" ;;
        right_3) echo "right_shift_3m" ;;
        *)
            echo "ERROR: invalid shift token '${1}'. Allowed: left_1|left_2|left_3|right_1|right_2|right_3" >&2
            return 1
            ;;
    esac
}

# 输入 CSV，输出全局数组 traj_types / distill_stage_repeats
build_difix_progressive_schedule() {
    local shifts_csv="$1"
    local novel_repeat="${2:-3}"
    local original_repeat="${3:-6}"

    traj_types=()
    distill_stage_repeats=()

    if [ -z "${shifts_csv}" ]; then
        echo "ERROR: novel_traj_shifts is empty" >&2
        return 1
    fi

    local IFS=','
    local -a tokens=(${shifts_csv})
    local token traj_type

    for token in "${tokens[@]}"; do
        token="${token// /}"
        [ -z "${token}" ] && continue
        traj_type=$(_difix_shift_token_to_traj_type "${token}") || return 1
        traj_types+=("${traj_type}" "original_traj")
        distill_stage_repeats+=("${novel_repeat}" "${original_repeat}")
    done

    if [ ${#traj_types[@]} -eq 0 ]; then
        echo "ERROR: no valid shift tokens in '${shifts_csv}'" >&2
        return 1
    fi
}

# 根据 shifts CSV 生成 checkpoint / video 后缀
build_difix_ckpt_suffix() {
    local shifts_csv="$1"
    local novel_repeat="${2:-3}"
    local original_repeat="${3:-6}"

    local left_part="" right_part=""
    local IFS=','
    local -a tokens=(${shifts_csv})
    local token dir num

    for token in "${tokens[@]}"; do
        token="${token// /}"
        [ -z "${token}" ] && continue
        case "${token}" in
            left_*)  dir="left";  num="${token#left_}"  ;;
            right_*) dir="right"; num="${token#right_}" ;;
            *) echo "ERROR: invalid token '${token}'" >&2; return 1 ;;
        esac
        if [ "${dir}" = "left" ]; then
            left_part="${left_part}${num}"
        else
            right_part="${right_part}${num}"
        fi
    done

    local tag="difix_prog"
    [ -n "${left_part}" ]  && tag="${tag}_left${left_part}"
    [ -n "${right_part}" ] && tag="${tag}_right${right_part}"
    tag="${tag}_ori_${original_repeat}_novel_${novel_repeat}"
    echo "${tag}"
}
