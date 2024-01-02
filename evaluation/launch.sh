#!/bin/bash
#SBATCH --job-name=histogram
#SBATCH --array=0-9
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="NVIDIAA100-SXM4-80GB|NVIDIAA10080GBPCIe"
#SBATCH --output=./logs/output-%A_%a.out
#SBATCH --error=./logs/output_%A_%a.err

srun python evaluate_length_histogram.py --rank ${SLURM_ARRAY_TASK_ID}
