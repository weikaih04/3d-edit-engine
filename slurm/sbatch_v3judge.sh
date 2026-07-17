#!/bin/bash
#SBATCH --job-name=v3judge
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/v3judge_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
echo "=== node $(hostname) $(date +%T) ==="
if [ ! -x /tmp/blender-4.5.1-linux-x64/blender ]; then
  tar -xf $B/blender-4.5.1-linux-x64.tar.xz -C /tmp
fi
echo "=== renders + grids (out_pairs3) ==="
PAIRS_ROOT=$B/out_pairs3 python $B/render_pairs.py 16 > $B/v3_render.log 2>&1
PAIRS_ROOT=$B/out_pairs3 python $B/grid_pairs.py > $B/v3_grids.log 2>&1
tail -1 $B/v3_grids.log

echo "=== manifest + dual-blind judge ==="
python - <<'EOF'
import os, json, glob
B = "/fsx/hyperpod/weikaih_edit"
items = []
for mp in sorted(glob.glob(f"{B}/out_pairs3/*/*/meta.json")):
    d = os.path.dirname(mp)
    task, name = d.split('/')[-2], os.path.basename(d)
    m = json.load(open(mp))
    if 'turns' in m:
        instr = " THEN ".join(t['instruction'] for t in m['turns'])
    else:
        instr = m.get('instruction', '?')
    items.append({"key": f"{task}/{name}", "grid": f"{B}/out_pairs3/compare/{task}__{name}.jpg",
                  "instruction": instr})
json.dump(items, open(f"{B}/judge_v3_manifest.json", "w"), indent=1)
print(len(items), "items")
EOF
conda deactivate; conda activate vlmcap
export PATH=/fsx/sfr/weikaih/miniconda3/envs/vlmcap/bin:$PATH
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python $B/dual_blind_judge.py \
  $B/judge_v3_manifest.json $B/judge_v3_results.json > $B/v3_judge.log 2>&1
tail -1 $B/v3_judge.log
echo "V3JUDGE_JOB_DONE $(date +%T)"
