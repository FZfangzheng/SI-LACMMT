#!/usr/bin/env bash

set -x

NAME='ast_photo2art_paris_0622'
TASK='AST'
DATA='photo2art'
#CROOT='/data3/fangzheng/map_generator/experiment/liangshuaizhe/about_Paris18/use_4900_norepaint/use'
#SROOT='/data3/fangzheng/map_generator/experiment/liangshuaizhe/about_Paris18/use_4900_norepaint/use'
CROOT='/data/fine_grained_multimap/dataset/liangshuaizhe/about_Paris18/use_4900_norepaint/use'
SROOT='/data/fine_grained_multimap/dataset/liangshuaizhe/about_Paris18/use_4900_norepaint/use'
CKPTROOT='./checkpoints'
WORKER=4
RESROOT='./results'
EPOCH='latest'

python test_liangshuaizhe.py \
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
    --how_many 10000 \
    --show_input
