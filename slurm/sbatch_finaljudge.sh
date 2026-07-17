#!/bin/bash
#SBATCH --job-name=finaljudge
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/finaljudge_%j.log

set -uo pipefail
B=/fsx/hyperpod/weikaih_edit
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
echo "=== node $(hostname) $(date +%T) ==="
python $B/build_judge_manifest.py round2 $B/judge_round2_manifest.json
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
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python $B/dual_blind_judge.py $B/judge_round2_manifest.json $B/judge_round2_results.json > $B/fj_r2.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python $B/dual_blind_judge.py $B/judge_v3_manifest.json $B/judge_v3_results.json > $B/fj_v3.log 2>&1 &
P2=$!
wait $P1; tail -1 $B/fj_r2.log
wait $P2; tail -1 $B/fj_v3.log
echo "FINALJUDGE_JOB_DONE $(date +%T)"
