#!/bin/bash

function pick_gpu() {
  local free_gpu
  free_gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '{print NR-1, $1}' | sort -nk2 | head -n1 | awk '{print $1}')
  echo ${free_gpu}
}
