#!/bin/bash
# Robust wrapper: relaunch pilot_texture on crash; blacklist the asset that
# was in-flight when the process died (dir exists but no meta.json).
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1 OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/fsx/hyperpod/weikaih_edit/TRELLIS.2
OUT=/fsx/hyperpod/weikaih_edit/out_pilot/null
cd /fsx/hyperpod/weikaih_edit/TRELLIS.2

for round in $(seq 1 30); do
  # blacklist any in-flight asset from a previous crash
  for d in $OUT/*/; do
    if [ ! -f "$d/meta.json" ]; then
      sha=$(basename "$d")
      echo "{\"sha\": \"$sha\", \"mode\": \"null\", \"error\": true, \"exc\": \"process died during this asset (blacklisted by wrapper)\"}" > "$d/meta.json"
      echo "[wrapper] blacklisted $sha"
    fi
  done
  n=$(ls $OUT/*/meta.json 2>/dev/null | wc -l)
  echo "[wrapper] round $round: $n/200 done"
  [ "$n" -ge 200 ] && break
  python /fsx/hyperpod/weikaih_edit/pilot_texture.py null 0 1 && break
  echo "[wrapper] python exited nonzero/crashed, restarting"
done
echo "NULL_ROBUST_DONE $(date +%T)"
