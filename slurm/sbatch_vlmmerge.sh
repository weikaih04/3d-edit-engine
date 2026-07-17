#!/bin/bash
#SBATCH --job-name=vlmmerge
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/vlmmerge_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
echo "=== node $(hostname) $(date +%T) ==="
CUDA_VISIBLE_DEVICES=0 python $B/vlm_merge_ask.py \
  200ecacfb41a4a5abdf819f68e5e5e8b 2ef12f2c94c246eca1b260d22c2a9c56 \
  6272df54d0cc4a318cc7d346d72da05b 88b61dbd2a37492a9301b2830df199e7 \
  802b1094c30d45d88df686cf20b197fb a1b5afe1f4dc47828eb0c0c5b9314a66
echo "VLMMERGE_JOB_DONE $(date +%T)"
