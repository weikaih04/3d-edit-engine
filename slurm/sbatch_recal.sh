#!/bin/bash
#SBATCH --job-name=judgecal
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=00:40:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/recal_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
echo "=== node $(hostname) $(date +%T) ==="
CUDA_VISIBLE_DEVICES=0 python $B/dual_blind_judge.py \
  $B/judge_round1_manifest.json $B/judge_round1_blind_results.json
echo "RECAL_JOB_DONE $(date +%T)"
