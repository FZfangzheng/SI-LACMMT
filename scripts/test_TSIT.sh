#!/usr/bin/env bash

set -x

NAME='TSIT'
TASK='AST'
DATA='photo2art'
CROOT='./dataset/CN_dataset'
SROOT='./dataset/CN_dataset'
CKPTROOT='./checkpoints'
WORKER=4
RESROOT='./results'
EPOCH='latest'

python ./src/TSIT/test.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --no_pairing_check \
    --no_instance \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --results_dir $RESROOT \
    --which_epoch $EPOCH \
    --show_input