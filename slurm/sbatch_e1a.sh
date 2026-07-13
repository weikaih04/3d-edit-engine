#!/bin/bash
#SBATCH --job-name=e1apilot
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/e1a_job_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
echo "=== node $(hostname) $(date +%T) ==="

# blender for step 3 (node-local /tmp; compute nodes have no internet)
if [ ! -x /tmp/blender-4.5.1-linux-x64/blender ]; then
  tar -xf $B/blender-4.5.1-linux-x64.tar.xz -C /tmp &
  BLTAR=$!
fi

echo "=== step 1: QIE-2511 edits, 8-way ==="
export HF_HOME=/fsx/sfr/weikaih/hf_cache HF_HUB_OFFLINE=1
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python $B/edit_views.py $B/e1a_instructions.json $g 8 \
    > $B/e1a_edit_g$g.log 2>&1 &
done
wait
grep -h "EDIT_VIEWS_DONE" $B/e1a_edit_g*.log | wc -l
ls $B/edited_views/*.png 2>/dev/null | wc -l

echo "=== step 2: texturing on edited views, 8-way ==="
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1 OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONPATH=$B/TRELLIS.2
cd $B/TRELLIS.2
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python $B/pilot_texture.py edit $g 8 \
    > $B/e1a_tex_g$g.log 2>&1 &
done
wait
ls $B/out_pilot/edit/*/after.glb 2>/dev/null | wc -l

echo "=== step 3: blender CPU renders + grids ==="
[ -n "${BLTAR:-}" ] && wait $BLTAR || true
python $B/render_outputs.py $B/out_pilot/edit 12 > $B/e1a_render.log 2>&1
tail -2 $B/e1a_render.log
echo "E1A_JOB_DONE $(date +%T)"
