#!/bin/bash
#SBATCH --job-name=edround2
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/round2_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
echo "=== node $(hostname) $(date +%T) ==="
if [ ! -x /tmp/blender-4.5.1-linux-x64/blender ]; then
  tar -xf $B/blender-4.5.1-linux-x64.tar.xz -C /tmp &
  BLTAR=$!
fi

echo "=== step 1: part naming v2 (visible-point refs) + dual-blind calibration (round1) ==="
conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
PAIRS_ROOT=$B/out_pairs2 CUDA_VISIBLE_DEVICES=0 python $B/name_parts.py > $B/r2_naming.log 2>&1 &
NAME_PID=$!
CUDA_VISIBLE_DEVICES=1 python $B/dual_blind_judge.py $B/judge_round1_manifest.json $B/judge_round1_results.json > $B/r2_judge1.log 2>&1 &
JUDGE_PID=$!
wait $NAME_PID; tail -2 $B/r2_naming.log
wait $JUDGE_PID; tail -1 $B/r2_judge1.log
conda deactivate; conda activate trellis2

echo "=== step 2: renders (blender CPU, anchored) ==="
[ -n "${BLTAR:-}" ] && wait $BLTAR || true
PAIRS_ROOT=$B/out_pairs2 python $B/render_pairs.py 16 > $B/r2_render.log 2>&1
tail -1 $B/r2_render.log

echo "=== step 3: grids ==="
PAIRS_ROOT=$B/out_pairs2 python $B/grid_pairs.py > $B/r2_grids.log 2>&1
tail -1 $B/r2_grids.log

echo "=== step 4: dual-blind judge on round2 ==="
python $B/build_judge_manifest.py round2 $B/judge_round2_manifest.json
conda deactivate; conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
CUDA_VISIBLE_DEVICES=0 python $B/dual_blind_judge.py $B/judge_round2_manifest.json $B/judge_round2_results.json > $B/r2_judge2.log 2>&1
tail -1 $B/r2_judge2.log
echo "ROUND2_JOB_DONE $(date +%T)"
