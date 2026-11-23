#!/bin/bash
# Example SLURM script for GPU clusters. Set --account, --partition and -C to your own allocation.

#SBATCH --job-name=flips-inference
#SBATCH --output=runs_outputs/outputs/%x_%j.out
#SBATCH --error=runs_outputs/errors/%x_%j.out

#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00 # Default time (can be overridden in sbatch command)
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --cpus-per-task=8
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH -C <YOUR_GPU_CONSTRAINT>

# Load your environment
module purge
# module load <your CUDA/PyTorch modules>

# Get executable
EXECUTABLE="$1"
shift

if [[ -z "$EXECUTABLE" ]]; then
  echo "Error: an executable must be provided as the first argument."
  echo "Usage: $0 <executable> [additional arguments]"
  exit 1
fi

# Cache directories (set CACHE_ROOT to a filesystem with room for the model weights)
CACHE_ROOT="${SCRATCH:-$HOME}/.cache"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"

# vLLM worker processes are spawned
export VLLM_WORKER_MULTIPROC_METHOD=spawn


# Enable verbose execution
set -x

# Run the job
srun python "$EXECUTABLE" "$@"

# Print execution time
duration=$SECONDS
echo "$((duration / 60)) min $((duration % 60)) sec"

# Examples:
#
# sbatch --time=02:00:00 --gres=gpu:1 scripts/run_large_LLMs.slurm scripts/Run_Inferences.py --dataset FLiPS_ICML --gpu 1 --model Qwen/Qwen2-7B-Instruct --sub_run qwen2_7b
# sbatch --time=04:00:00 --gres=gpu:4 --cpus-per-task=32 scripts/run_large_LLMs.slurm scripts/Run_Inferences.py --dataset FLiPS_ICML --gpu 4 --model Qwen/Qwen2-72B-Instruct --sub_run qwen2_72b
