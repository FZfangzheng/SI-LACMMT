#!/usr/bin/env bash

set -x

NAME='ast_photo2art_pp'
TASK='AST'
DATA='photo2art'
CROOT='/data3/fangzheng/map_generator/experiment/SH/data_all_LAMG'
SROOT='/data3/fangzheng/map_generator/experiment/SH/data_all_LAMG'
CKPTROOT='./checkpoints'
WORKER=4
RESROOT='./results'
EPOCH='latest'

python test.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 2 \
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
