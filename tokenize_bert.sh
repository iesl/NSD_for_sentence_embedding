#!/bin/bash
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00
#SBATCH --mem=60000

~/anaconda3/bin/python ./src/preprocessing/tools/tokenize_bert.py   
