#!/bin/bash
#SBATCH --job-name=partname
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=00:30:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/naming_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
echo "=== node $(hostname) $(date +%T) ==="
CUDA_VISIBLE_DEVICES=0 python $B/name_parts.py
echo "NAMING_JOB_DONE $(date +%T)"
