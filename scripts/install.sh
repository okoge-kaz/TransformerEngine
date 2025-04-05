#!/bin/bash
#SBATCH --job-name=install
#SBATCH --partition=part-group_d2f6d7
#SBATCH --time=0-01:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/install/%x-%j.out
#SBATCH --error=outputs/install/%x-%j.out

set -e

module load python/3.12.7
module load cuda/12.8.1
module load cudnn/9.8.0.87_cuda12
module load nccl/2.26.2-1-cuda-12.8.1
module load hpcx/v2.18.1-cuda12

# srun --partition=part-group_d2f6d7 --nodes=1 --job-name=install --time=0-03:00:00 --pty bash

source .env/bin/activate

# pip install
pip install --upgrade pip
pip install --upgrade wheel cmake ninja

pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir

pip install pybind11

# transformer engine v2.2 or later
cudnn_root="/opt/share/modules/cudnn/9.8.0.87_cuda12"

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$cudnn_root/lib
LIBRARY_PATH=$LIBRARY_PATH:$cudnn_root/lib
CPATH=$CPATH:$cudnn_root/include

export CUDNN_PATH=$cudnn_root
export CUDNN_INCLUDE_DIR=$cudnn_root/include
export CUDNN_LIBRARY_DIR=$cudnn_root/lib
export CUDNN_ROOT_DIR=$cudnn_root

git clone https://github.com/NVIDIA/TransformerEngine.git
git submodule update --init --recursive

export NVTE_FRAMEWORK=pytorch

pip install -e .
