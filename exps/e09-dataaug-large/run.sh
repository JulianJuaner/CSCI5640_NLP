#!/bin/bash
#SBATCH --job-name=nlp-e09-dataaug-large
#SBATCH --cpus-per-task=6
#SBATCH --output=./exps/e09-dataaug-large/logger_debug.log
#SBATCH --gres=gpu:1

conda activate 3090
export PYTHONPATH=/research/d4/gds/yczhang21/project/CSCI5640_NLP/flair:$PYTHONPATH
python3.9 exps/e09-dataaug-large/trainval.py
