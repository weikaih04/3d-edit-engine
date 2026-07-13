#!/bin/bash
#SBATCH --job-name=e4e8chain
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=02:30:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/chain_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
echo "=== node $(hostname) $(date +%T) ==="
if [ ! -x /tmp/blender-4.5.1-linux-x64/blender ]; then
  tar -xf $B/blender-4.5.1-linux-x64.tar.xz -C /tmp &
  BLTAR=$!
fi

echo "=== step 1: X-Part geometry re-synthesis (E4 + E8, 2 GPUs) ==="
conda activate p3sam
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
export HY3DGEN_MODELS=/fsx/hyperpod/xpart_models
cd $B/Hunyuan3D-Part/XPart
CUDA_VISIBLE_DEVICES=0 python $B/xpart_chain_geo.py E4 8 > $B/ch_geo_e4.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python $B/xpart_chain_geo.py E8 8 > $B/ch_geo_e8.log 2>&1 &
P2=$!
wait $P1; tail -1 $B/ch_geo_e4.log
wait $P2; tail -1 $B/ch_geo_e8.log
conda deactivate

echo "=== step 2: TRELLIS.2 re-texture, 8-way ==="
conda activate trellis2
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1 OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONPATH=$B/TRELLIS.2
cd $B/TRELLIS.2
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python $B/xpart_chain_tex.py E4 $g 4 > $B/ch_tex_e4_$g.log 2>&1 &
done
for g in 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python $B/xpart_chain_tex.py E8 $((g-4)) 4 > $B/ch_tex_e8_$g.log 2>&1 &
done
wait
echo "tex: E4=$(ls $B/out_pairs2/E4/*/after.glb 2>/dev/null | wc -l) E8=$(ls $B/out_pairs2/E8/*/after.glb 2>/dev/null | wc -l)"

echo "=== step 3: renders + zoom + grids ==="
[ -n "${BLTAR:-}" ] && wait $BLTAR || true
PAIRS_ROOT=$B/out_pairs2 python $B/render_pairs.py 16 > $B/ch_render.log 2>&1
python $B/render_zoom.py 16 > $B/ch_zoom.log 2>&1 || true
PAIRS_ROOT=$B/out_pairs2 python $B/grid_pairs.py > $B/ch_grids.log 2>&1
tail -1 $B/ch_grids.log

echo "=== step 4: full round2 re-judge ==="
python $B/build_judge_manifest.py round2 $B/judge_round2_manifest.json
conda deactivate; conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
CUDA_VISIBLE_DEVICES=0 python $B/dual_blind_judge.py \
  $B/judge_round2_manifest.json $B/judge_round2_results.json > $B/ch_judge.log 2>&1
tail -1 $B/ch_judge.log
echo "CHAIN_JOB_DONE $(date +%T)"
