#!/bin/bash
#SBATCH --job-name=nlp-e08
#SBATCH --cpus-per-task=6
#SBATCH --output=./exps/e08-base-4cls/logger_debug.log
#SBATCH --gres=gpu:1

conda activate imatt
export PYTHONPATH=/research/d4/gds/yczhang21/project/CSCI5640_NLP/flair:$PYTHONPATH
python3.9 exps/e08-base-4cls/trainval.py
