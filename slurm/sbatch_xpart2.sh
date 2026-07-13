#!/bin/bash
#SBATCH --job-name=xpart2
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/xpart2_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
echo "=== node $(hostname) $(date +%T) ==="

echo "=== X-Part re-synthesis test ==="
conda activate p3sam
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
export HY3DGEN_MODELS=/fsx/hyperpod/xpart_models
cd $B/Hunyuan3D-Part/XPart
CUDA_VISIBLE_DEVICES=0 python $B/xpart_test.py 3
conda deactivate

echo "=== refresh round2 judge (incl E1b v2) ==="
conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
python $B/build_judge_manifest.py round2 $B/judge_round2_manifest.json
CUDA_VISIBLE_DEVICES=1 python $B/dual_blind_judge.py \
  $B/judge_round2_manifest.json $B/judge_round2_results.json > $B/xp2_judge.log 2>&1
tail -1 $B/xp2_judge.log
echo "XPART2_JOB_DONE $(date +%T)"
