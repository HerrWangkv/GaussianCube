#!/bin/bash
#SBATCH --job-name=gaussiancube_vehicle
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/finetune_vehicle_%j.out
#SBATCH --error=logs/finetune_vehicle_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting time: $(date)"
echo "Working directory: $(pwd)"

# Create necessary directories for MPI and caching
mkdir -p .cache/.config/matplotlib
mkdir -p .cache/mpi_tmp

# Set MPI environment variables to avoid /scratch directory issues
export TMPDIR=$(pwd)/.cache/mpi_tmp
export ORTE_TMPDIR_BASE=$(pwd)/.cache/mpi_tmp
export OMPI_MCA_orte_tmpdir_base=$(pwd)/.cache/mpi_tmp
# Run the container and execute the finetune script
echo "Starting apptainer container from gaussiancube.sif..."

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    --env XDG_CACHE_HOME=/workspace/.cache \
    --env MPLCONFIGDIR=/workspace/.cache/.config/matplotlib \
    --env TMPDIR=/workspace/.cache/mpi_tmp \
    --env ORTE_TMPDIR_BASE=/workspace/.cache/mpi_tmp \
    --env OMPI_MCA_orte_tmpdir_base=/workspace/.cache/mpi_tmp \
    --env HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN} \
    gaussiancube.sif \
    bash finetune_vehicle.sh

echo "Job completed at: $(date)"
