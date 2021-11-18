#!/bin/bash
#SBATCH --job-name=simclr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH -o %j.out
#SBATCH -e %j.out

DATASET=CIFAR10
BATCH_SIZE=384
LOSS="dclw"
TEMP=0.4
EPOCHS=20

python train.py \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --feature_dim 128 \
  --loss $LOSS \
  --temperature $TEMP \
  --rank 0 \
  --world-size 1 \
  --multiprocessing-distributed \


python test.py \
  --batch_size 384 \
  --epochs 20 \
  --model_path "results/128_${TEMP}_200_${BATCH_SIZE}_${EPOCHS}_${LOSS}_model.pth"
