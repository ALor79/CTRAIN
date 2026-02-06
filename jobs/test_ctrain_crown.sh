#!/usr/bin/zsh
############################################################
### Slurm flags
############################################################

#SBATCH --partition=c23g            # request partition with GPU nodes
#SBATCH --nodes=1                   # request desired number of nodes
#SBATCH --ntasks-per-node=1         # request desired number of processes (or MPI tasks)

#SBATCH --cpus-per-task=24          # request desired number of CPU cores or threads per process (default: 1)
                                    # Note: available main memory is also scaling with
                                    #       number of cores if not specified otherwise
                                    # Note: On CLAIX-2023 each GPU can be used with 24 cores

#SBATCH --gres=gpu:1                # specify desired number of GPUs per node
#SBATCH --time=08:00:00             # max. run time of the job
#SBATCH --job-name=CONVEX_HULL_PGD   # set the job name
#SBATCH --output=logs/ctrain_test_%j.log
#SBATCH --error=logs/ctrain_test_%j.err

############################################################
### Parameters and Settings
############################################################

# print some information about current system
echo "Job nodes: ${SLURM_JOB_NODELIST}"
echo "Current machine: $(hostname)"
nvidia-smi


module load GCCcore/.13.2.0
module load Python/3.11.5 
module load CUDA/12.3.0


source ~/dev/CTRAIN/venv/bin/activate

cd ~/dev/CTRAIN
python3 test_ctrain_crown_cifar_HPO.py