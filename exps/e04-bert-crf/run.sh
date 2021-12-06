#!/bin/bash
#SBATCH --job-name=nlp-e02
#SBATCH --mail-user=zhangyc@link.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=10
#SBATCH --output=./exps/e04-bert-crf/logger_debug.log
#SBATCH --gres=gpu:2

conda activate 3090
export PYTHONPATH=/research/d4/gds/yczhang21/project/CSCI5640_NLP/flair:$PYTHONPATH
python3.9 exps/e04-bert-crf/trainval.py
