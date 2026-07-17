#!/bin/bash
#SBATCH --job-name=e1bv3
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/e1bv3_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
echo "=== node $(hostname) $(date +%T) ==="
if [ ! -x /tmp/blender-4.5.1-linux-x64/blender ]; then
  tar -xf $B/blender-4.5.1-linux-x64.tar.xz -C /tmp &
  BLTAR=$!
fi

echo "=== step 1: QIE targeted edits (semantic units), 8-way ==="
export HF_HOME=/fsx/sfr/weikaih/hf_cache HF_HUB_OFFLINE=1
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python $B/edit_views_multi.py $g 8 \
    $B/e1b_v3_instructions.json:$B/e1b_v3_views > $B/ev3_edit_g$g.log 2>&1 &
done
wait
echo "views: $(ls $B/e1b_v3_views/*.png 2>/dev/null | wc -l)"

echo "=== step 2: texturing, 8-way ==="
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1 OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONPATH=$B/TRELLIS.2
cd $B/TRELLIS.2
for g in 0 1 2 3 4 5 6 7; do
  EDITED_DIR=$B/e1b_v3_views PILOT_OUT=$B/out_pilot/e1b_v3 \
    CUDA_VISIBLE_DEVICES=$g python $B/pilot_texture.py edit $g 8 > $B/ev3_tex_g$g.log 2>&1 &
done
wait
echo "tex: $(ls $B/out_pilot/e1b_v3/*/after.glb 2>/dev/null | wc -l)"

echo "=== step 3: unit-mask merge with REBAKED outside (CPU) ==="
cd $B
rm -rf $B/out_pairs3/E1b
E1B_EDIT_DIR=$B/out_pilot/e1b_v3 E1B_VIEWS_DIR=$B/e1b_v3_views \
  NULL_DIR=$B/out_pilot/null_v2 E1B_OUT=$B/out_pairs3/E1b \
  E1B_PART_MAP=$B/e1b_v3_parts.json E1B_OUTSIDE=rebake \
  python $B/e1b_merge.py 12 > $B/ev3_merge.log 2>&1
tail -1 $B/ev3_merge.log

echo "=== step 4: renders + grids + judge refresh ==="
[ -n "${BLTAR:-}" ] && wait $BLTAR || true
PAIRS_ROOT=$B/out_pairs3 python $B/render_pairs.py 16 > $B/ev3_render.log 2>&1
PAIRS_ROOT=$B/out_pairs3 python $B/grid_pairs.py > $B/ev3_grids.log 2>&1
python - <<'EOF'
import os, json, glob
B = "/fsx/hyperpod/weikaih_edit"
items = []
for mp in sorted(glob.glob(f"{B}/out_pairs3/*/*/meta.json")):
    d = os.path.dirname(mp)
    task, name = d.split('/')[-2], os.path.basename(d)
    m = json.load(open(mp))
    instr = " THEN ".join(t['instruction'] for t in m['turns']) if 'turns' in m else m.get('instruction', '?')
    items.append({"key": f"{task}/{name}", "grid": f"{B}/out_pairs3/compare/{task}__{name}.jpg",
                  "instruction": instr})
json.dump(items, open(f"{B}/judge_v3_manifest.json", "w"), indent=1)
print(len(items), "v3 items")
EOF
conda deactivate; conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
export HF_HOME=/fsx/hyperpod/hf_cache
CUDA_VISIBLE_DEVICES=0 python $B/dual_blind_judge.py \
  $B/judge_v3_manifest.json $B/judge_v3_results.json > $B/ev3_judge.log 2>&1
tail -1 $B/ev3_judge.log
echo "E1BV3_JOB_DONE $(date +%T)"
