#!/bin/bash
#SBATCH --job-name=nlp-e01
#SBATCH --mail-user=zhangyc@link.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --constrain=rtx2080
conda activate imatt
export PYTHONPATH=/research/d4/gds/yczhang21/project/CSCI5640_NLP/:$PYTHONPATH
python3.9 exps/e01-base/trainval.py
