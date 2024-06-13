#!/bin/bash
#SBATCH --job-name=generate_fake_true_dataset_gemma
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=0-05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/dash/text_llm_detector
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Activate environment
eval "$(conda shell.bash hook)"
conda activate llm_detector

python src/generate_fake_true_dataset.py --generator=gemma_2b --batch_size=64 --experiment_name=gemma_10k --fake_dataset_size=10000 --access_token=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
conda deactivate
