#!/bin/bash
# E1a pilot chain: edit 30 views with QIE-2511, then run texturing on edited views
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/fsx/sfr/weikaih/hf_cache HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12

echo "=== step 1: QIE-2511 edits ==="
python /fsx/hyperpod/weikaih_edit/edit_views.py /fsx/hyperpod/weikaih_edit/e1a_instructions.json 30 \
  || { echo "EDIT_STEP_FAILED"; exit 1; }

echo "=== step 2: texturing with edited views ==="
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1 OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONPATH=/fsx/hyperpod/weikaih_edit/TRELLIS.2
cd /fsx/hyperpod/weikaih_edit/TRELLIS.2
python /fsx/hyperpod/weikaih_edit/pilot_texture.py edit 0 1
echo "E1A_CHAIN_DONE $(date +%T)"
