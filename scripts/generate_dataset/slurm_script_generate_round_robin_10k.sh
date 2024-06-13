#!/bin/bash
#SBATCH --job-name=generate_fake_true_dataset
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=nodes
#SBATCH --gres=gpu:titanv:1
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

# note: batch size 512 is too much
python src/generate_round_robin_dataset.py --save_dir=fake_true_datasets
conda deactivate
