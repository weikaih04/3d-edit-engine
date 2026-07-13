#!/bin/bash
#SBATCH --job-name=e1bv2
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72
#SBATCH --time=02:00:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/e1bv2_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
echo "=== node $(hostname) $(date +%T) ==="
if [ ! -x /tmp/blender-4.5.1-linux-x64/blender ]; then
  tar -xf $B/blender-4.5.1-linux-x64.tar.xz -C /tmp &
  BLTAR=$!
fi

echo "=== step 1: name selected parts ==="
conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python $B/name_refs.py $B/e1b_v2_refs $B/e1b_v2_names.json > $B/ev2_naming.log 2>&1
tail -1 $B/ev2_naming.log

echo "=== step 2: targeted instructions ==="
python - <<'EOF'
import json
B = "/fsx/hyperpod/weikaih_edit"
names = json.load(open(f"{B}/e1b_v2_names.json"))
mats = ["polished gold metal", "red leather", "carved dark walnut wood",
        "white marble with gray veins", "brushed stainless steel",
        "glossy blue ceramic", "weathered rusty iron", "translucent green jade",
        "matte black carbon fiber", "bright red glossy paint"]
out = {}
for i, (sha, nm) in enumerate(sorted(names.items())):
    out[sha] = (f"Change only the {nm} to {mats[i % len(mats)]}. Keep every other part "
                f"and the shape, pose, silhouette and background exactly the same.")
json.dump(out, open(f"{B}/e1b_v2_instructions.json", "w"), indent=1)
print(len(out), "targeted instructions")
EOF
conda deactivate; conda activate trellis2

echo "=== step 3: QIE targeted edits, 8-way ==="
export HF_HOME=/fsx/sfr/weikaih/hf_cache HF_HUB_OFFLINE=1
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python $B/edit_views_multi.py $g 8 \
    $B/e1b_v2_instructions.json:$B/e1b_views > $B/ev2_edit_g$g.log 2>&1 &
done
wait
echo "views: $(ls $B/e1b_views/*.png 2>/dev/null | wc -l)"

echo "=== step 4: texturing, 8-way ==="
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1 OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONPATH=$B/TRELLIS.2
cd $B/TRELLIS.2
for g in 0 1 2 3 4 5 6 7; do
  EDITED_DIR=$B/e1b_views PILOT_OUT=$B/out_pilot/e1b_v2 \
    CUDA_VISIBLE_DEVICES=$g python $B/pilot_texture.py edit $g 8 > $B/ev2_tex_g$g.log 2>&1 &
done
wait
echo "tex: $(ls $B/out_pilot/e1b_v2/*/after.glb 2>/dev/null | wc -l)"

echo "=== step 5: masked merge (CPU) ==="
rm -rf $B/out_pairs2/E1b
E1B_EDIT_DIR=$B/out_pilot/e1b_v2 E1B_VIEWS_DIR=$B/e1b_views \
  NULL_DIR=$B/out_pilot/null_v2 E1B_OUT=$B/out_pairs2/E1b \
  E1B_PART_MAP=$B/e1b_v2_parts.json \
  python $B/e1b_merge.py 12 > $B/ev2_merge.log 2>&1
tail -1 $B/ev2_merge.log

echo "=== step 6: renders + grids ==="
[ -n "${BLTAR:-}" ] && wait $BLTAR || true
PAIRS_ROOT=$B/out_pairs2 python $B/render_pairs.py 16 > $B/ev2_render.log 2>&1
PAIRS_ROOT=$B/out_pairs2 python $B/grid_pairs.py > $B/ev2_grids.log 2>&1
tail -1 $B/ev2_grids.log

echo "=== step 7: refresh round2 judge ==="
python $B/build_judge_manifest.py round2 $B/judge_round2_manifest.json
conda deactivate; conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
CUDA_VISIBLE_DEVICES=0 python $B/dual_blind_judge.py \
  $B/judge_round2_manifest.json $B/judge_round2_results.json > $B/ev2_judge.log 2>&1
tail -1 $B/ev2_judge.log
echo "E1BV2_JOB_DONE $(date +%T)"
