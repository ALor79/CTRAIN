echo "Current machine: $(hostname)"
nvidia-smi

# module load GCCcore/.13.2.0
# module load Python/3.11.5
# module load CUDA/12.3.0

source ~/dev/CTRAIN/venv/bin/activate

cd ~/dev/CTRAIN
export PYTHONWARNINGS="ignore"
python -W ignore tests/test_diff_ai_table2_cifar.py > >(tee logs/diff_ai_table2_$$.log) 2> >(tee ./logs/errors/diff_ai_table2_$$.err >&2)
