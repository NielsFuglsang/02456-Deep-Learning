#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J name
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/name.out
#BSUB -e logs/name.err

cd /zhome/ff/2/118359/projects/02456-Deep-Learning
source .venv/bin/activate

echo "Running script..."
python run_experiment.py name
