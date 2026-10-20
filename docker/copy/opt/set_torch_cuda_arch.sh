#!/bin/bash

arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u 2>/dev/null | tr '\n' ';' | sed 's/;$//')

if [ -z "$arch" ]; then
    arch=""
fi

if ! grep -q "TORCH_CUDA_ARCH_LIST" /etc/profile; then
    echo "export TORCH_CUDA_ARCH_LIST=\"$arch\"" >> /etc/profile
    echo "TORCH_CUDA_ARCH_LIST set to $arch in /etc/profile"
else
    echo "TORCH_CUDA_ARCH_LIST already set in /etc/profile"
fi
