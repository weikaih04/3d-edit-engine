#!/bin/bash
#SBATCH --job-name=e5fix
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/e5fix_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
echo "=== node $(hostname) $(date +%T) ==="
if [ ! -x /tmp/blender-4.5.1-linux-x64/blender ]; then
  tar -xf $B/blender-4.5.1-linux-x64.tar.xz -C /tmp &
  BLTAR=$!
fi
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1 OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONPATH=$B/TRELLIS.2
cd $B/TRELLIS.2

echo "=== E5 full regen, 8-way ==="
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python $B/pilot_regen.py $g 8 > $B/e5fix_g$g.log 2>&1 &
done
wait
echo "e5 done: $(ls $B/out_pilot/e5/*/after.glb 2>/dev/null | wc -l)"

echo "=== renders + grids ==="
[ -n "${BLTAR:-}" ] && wait $BLTAR || true
python $B/render_outputs.py $B/out_pilot/e5 8 > $B/e5fix_render.log 2>&1
G_EDIT_DIR=$B/out_pilot/e5 G_VIEWS_DIR=$B/e5_views \
  G_OUT=$B/out_pilot/compare_e5 G_TAG=E5 python $B/build_e1a_grids.py
echo "E5FIX_DONE $(date +%T)"
