#!/bin/bash
# Activate venv with GPU support for TensorFlow 2.10.1
# Usage: source activate_gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Set LD_LIBRARY_PATH for pip-installed CUDA libraries
export LD_LIBRARY_PATH="$(python -c "
import nvidia.cudnn, nvidia.cublas, nvidia.cuda_runtime, nvidia.cufft, nvidia.curand, nvidia.cusolver, nvidia.cusparse
paths = [
    nvidia.cudnn.__path__[0] + '/lib',
    nvidia.cublas.__path__[0] + '/lib',
    nvidia.cuda_runtime.__path__[0] + '/lib',
    nvidia.cufft.__path__[0] + '/lib',
    nvidia.curand.__path__[0] + '/lib',
    nvidia.cusolver.__path__[0] + '/lib',
    nvidia.cusparse.__path__[0] + '/lib',
]
print(':'.join(paths))
"):$LD_LIBRARY_PATH"

# Add ptxas to PATH for XLA compilation
NVCC_PATH="$(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__path__[0])")"
export PATH="$NVCC_PATH/bin:$PATH"

# Set XLA CUDA data dir to the nvcc package (contains libdevice)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVCC_PATH"

echo "Activated venv with GPU support"
echo "Python: $(python --version)"
echo "TensorFlow GPU check:"
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'  GPUs: {gpus}')" 2>/dev/null
