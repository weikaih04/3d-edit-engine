#!/bin/bash
#SBATCH --job-name=edtpilot
#SBATCH --account=low-pri
#SBATCH --requeue
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --exclude=ip-10-0-155-72,ip-10-0-134-150
#SBATCH --time=8:00:00
#SBATCH --output=/fsx/hyperpod/weikaih_edit/slurm_%x_%j.out
# Pilot: TRELLIS.2 texturing null-edit fidelity on 200 assets, 8-way data parallel.
# Usage: sbatch sbatch_pilot.sh <mode> [limit_per_shard]
set -e
MODE=${1:-null}
LIMIT=${2:-1000}
srun --kill-on-bad-exit=0 bash -c "
source /fsx/sfr/weikaih/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
export HF_HOME=/fsx/hyperpod/hf_cache HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
export PYTHONPATH=/fsx/hyperpod/weikaih_edit/TRELLIS.2:\$PYTHONPATH
cd /fsx/hyperpod/weikaih_edit/TRELLIS.2
python /fsx/hyperpod/weikaih_edit/pilot_texture.py $MODE \$SLURM_LOCALID 8 $LIMIT \
  > /fsx/hyperpod/weikaih_edit/pilot_${MODE}_shard\${SLURM_LOCALID}.log 2>&1
"
echo "PILOT_DONE $MODE $(date +%T)"
