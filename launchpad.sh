#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"


module load anaconda

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

conda activate toy_classification_env # open the Python environment

pip install -r requirements.txt # install Python dependencies

# runs your code
srun python classification.py  --experiment "overfit" --small_subset --device cuda --model "distilbert-base-uncased" --batch_size "32" --lr 1e-4 --num_epochs 20


# runs your code
# use a subset of data
srun python classification.py  --experiment "overfit" --small_subset --device cuda --model "distilbert-base-uncased" --batch_size "32" --lr 1e-4 --num_epochs 20

# run on full dataset
srun python classification.py  --experiment "1-dbu-e30" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 30

# hyper-parameter selection
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 5
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 7
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 9
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 5e-4 --num_epochs 5
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 5e-4 --num_epochs 7
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 5e-4 --num_epochs 9
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-3 --num_epochs 5
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-3 --num_epochs 7
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-3 --num_epochs 9

# largest batch size 64
srun python classification.py  --experiment "1-dbu" --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 9

# largest batch size 32
srun python classification.py  --experiment "2-bbu" --device cuda --model "bert-base-uncased" --batch_size "32" --lr 1e-4 --num_epochs 9
srun python classification.py  --experiment "4-bbc" --device cuda --model "bert-base-cased" --batch_size "32" --lr 1e-4 --num_epochs 9
srun python classification.py  --experiment "6-rb" --device cuda --model "roberta-base" --batch_size "32" --lr 1e-4 --num_epochs 9

# largest batch size 8
srun python classification.py  --experiment "3-blu" --device cuda --model "bert-large-uncased" --batch_size "8" --lr 1e-4 --num_epochs 9
srun python classification.py  --experiment "5-blc" --device cuda --model "bert-large-cased" --batch_size "8" --lr 1e-4 --num_epochs 9

# largest batch size 4
srun python classification.py  --experiment "7-rl" --device cuda --model "roberta-large" --batch_size "4" --lr 1e-4 --num_epochs 9