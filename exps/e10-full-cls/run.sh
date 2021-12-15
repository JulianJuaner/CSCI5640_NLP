#!/bin/bash
#SBATCH --job-name=nlp-e10-full
#SBATCH --mail-user=zhangyc@link.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1

conda activate imatt
export PYTHONPATH=/research/d4/gds/yczhang21/project/CSCI5640_NLP/flair:$PYTHONPATH
python3.9 exps/e10-full-cls/trainval.py
