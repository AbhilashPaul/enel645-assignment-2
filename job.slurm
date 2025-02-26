#!/bin/bash
#SBATCH --job-name=enel645-a2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=stdout_%j.out
#SBATCH --error=stderror_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB

source ~/software/init-conda.sh
conda activate enel645

python assignment2.py

conda deactivate