#!/usr/bin/zsh
############################################################
### Slurm flags
############################################################

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=CTRAIN_GOWAL_SMALL_IBP
#SBATCH --output=logs/ctrain_gowal_small_ibp_%j.log
#SBATCH --error=logs/ctrain_gowal_small_ibp_%j.err

############################################################
### Parameters and Settings
############################################################

echo "Job nodes: ${SLURM_JOB_NODELIST}"
echo "Current machine: $(hostname)"
nvidia-smi

module load GCCcore/.13.2.0
module load Python/3.11.5
module load CUDA/12.3.0

source ~/dev/CTRAIN/venv/bin/activate

cd ~/dev/CTRAIN
export PYTHONWARNINGS="ignore"
python -W ignore tests/test_ctrain_cifar10_gowal_small_ibp.py
