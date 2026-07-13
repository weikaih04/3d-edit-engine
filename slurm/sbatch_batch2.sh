#!/bin/bash
#SBATCH --job-name=e12batch2
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=03:00:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/batch2_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
echo "=== node $(hostname) $(date +%T) ==="

if [ ! -x /tmp/blender-4.5.1-linux-x64/blender ]; then
  tar -xf $B/blender-4.5.1-linux-x64.tar.xz -C /tmp &
  BLTAR=$!
fi

echo "=== step 1: QIE edits (E1a-v2 good views + E1c + E5), 8-way ==="
export HF_HOME=/fsx/sfr/weikaih/hf_cache HF_HUB_OFFLINE=1
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python $B/edit_views_multi.py $g 8 \
    $B/e1a_instructions.json:$B/edited_views_v2 \
    $B/e1c_instructions.json:$B/e1c_views \
    $B/e5_instructions.json:$B/e5_views \
    > $B/b2_edit_g$g.log 2>&1 &
done
wait
echo "views: e1a_v2=$(ls $B/edited_views_v2/*.png 2>/dev/null | wc -l) e1c=$(ls $B/e1c_views/*.png 2>/dev/null | wc -l) e5=$(ls $B/e5_views/*.png 2>/dev/null | wc -l)"

export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1 OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONPATH=$B/TRELLIS.2
cd $B/TRELLIS.2

echo "=== step 2a: texturing E1a-v2 + null-v2 (good views) 8-way ==="
for g in 0 1 2 3 4 5 6 7; do
  ( EDITED_DIR=$B/edited_views_v2 PILOT_OUT=$B/out_pilot/edit_v2 \
      CUDA_VISIBLE_DEVICES=$g python $B/pilot_texture.py edit $g 8 ;
    EDITED_DIR=$B/edited_views_v2 PILOT_OUT=$B/out_pilot/null_v2 ONLY_EDITED=1 \
      CUDA_VISIBLE_DEVICES=$g python $B/pilot_texture.py null $g 8 ;
    EDITED_DIR=$B/e1c_views PILOT_OUT=$B/out_pilot/e1c \
      CUDA_VISIBLE_DEVICES=$g python $B/pilot_texture.py edit $g 8 ;
    CUDA_VISIBLE_DEVICES=$g python $B/pilot_regen.py $g 8 ) \
    > $B/b2_tex_g$g.log 2>&1 &
done
wait
echo "tex done: edit_v2=$(ls $B/out_pilot/edit_v2/*/after.glb 2>/dev/null | wc -l) null_v2=$(ls $B/out_pilot/null_v2/*/after.glb 2>/dev/null | wc -l) e1c=$(ls $B/out_pilot/e1c/*/after.glb 2>/dev/null | wc -l) e5=$(ls $B/out_pilot/e5/*/after.glb 2>/dev/null | wc -l)"

echo "=== step 3: E1b re-merge on v2 (CPU) ==="
rm -rf $B/out_pairs/E1b
E1B_EDIT_DIR=$B/out_pilot/edit_v2 E1B_VIEWS_DIR=$B/edited_views_v2 \
  NULL_DIR=$B/out_pilot/null_v2 \
  python $B/e1b_merge.py 10 > $B/b2_e1b.log 2>&1
tail -1 $B/b2_e1b.log

echo "=== step 4: part naming (Qwen3.6-27B, GPU0) ==="
conda deactivate; conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
CUDA_VISIBLE_DEVICES=0 HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1 \
  python $B/name_parts.py > $B/b2_naming.log 2>&1
tail -2 $B/b2_naming.log
conda deactivate; conda activate trellis2

echo "=== step 5: renders (blender CPU) ==="
[ -n "${BLTAR:-}" ] && wait $BLTAR || true
python $B/render_pairs.py 16 > $B/b2_render_pairs.log 2>&1 &
RP=$!
python $B/render_outputs.py $B/out_pilot/edit_v2 8 > $B/b2_render_e1a.log 2>&1
python $B/render_outputs.py $B/out_pilot/e1c 8 > $B/b2_render_e1c.log 2>&1
python $B/render_outputs.py $B/out_pilot/e5 8 > $B/b2_render_e5.log 2>&1
wait $RP

echo "=== step 6: grids ==="
python $B/grid_pairs.py > $B/b2_grids.log 2>&1
G_EDIT_DIR=$B/out_pilot/edit_v2 G_VIEWS_DIR=$B/edited_views_v2 \
  G_OUT=$B/out_pilot/compare_e1a_v2 G_TAG=E1a python $B/build_e1a_grids.py >> $B/b2_grids.log 2>&1
G_EDIT_DIR=$B/out_pilot/e1c G_VIEWS_DIR=$B/e1c_views \
  G_OUT=$B/out_pilot/compare_e1c G_TAG=E1c python $B/build_e1a_grids.py >> $B/b2_grids.log 2>&1
G_EDIT_DIR=$B/out_pilot/e5 G_VIEWS_DIR=$B/e5_views \
  G_OUT=$B/out_pilot/compare_e5 G_TAG=E5 python $B/build_e1a_grids.py >> $B/b2_grids.log 2>&1
tail -4 $B/b2_grids.log
echo "BATCH2_DONE $(date +%T)"
