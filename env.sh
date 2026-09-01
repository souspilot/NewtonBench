#!/usr/bin/env bash
# Source this on a GPU compute node before serving/benchmarking:
#   source env.sh
module load compiler/cuda/12.3/compilervars 2>/dev/null
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export TRITON_PTXAS_PATH="$(which ptxas)"     # force Triton to use the module's ptxas
export JAX_PLATFORMS=cpu
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export no_proxy=localhost,127.0.0.1,::1,$(hostname)
export NO_PROXY="$no_proxy"

if command -v nvcc >/dev/null && command -v ptxas >/dev/null; then
  echo "CUDA ready: nvcc=$(which nvcc)  CUDA_HOME=$CUDA_HOME  host=$(hostname)"
else
  echo "WARNING: nvcc/ptxas not found — are you on a GPU node? (run nvidia-smi)"
fi
