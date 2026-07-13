#!/bin/bash
#SBATCH --job-name=xparttest
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=00:50:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/xpart_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate p3sam
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
export HY3DGEN_MODELS=/fsx/hyperpod/xpart_models
export CUDA_VISIBLE_DEVICES=0
echo "=== node $(hostname) $(date +%T) ==="
cd $B/Hunyuan3D-Part/XPart
python $B/xpart_test.py 3
echo "XPART_JOB_DONE $(date +%T)"
