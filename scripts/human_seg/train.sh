#!/usr/bin/env bash

## run the training
CUDA_VISIBLE_DEVICES=4 python train.py \
--dataroot /disk5/data/MeshCNN/datasets/human_seg \
--name human_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 2280 \
--pool_res 1800 1350 600 \
--resblocks 3 \
--batch_size 12 \
--lr 0.001 \
--num_aug 20 \
--slide_verts 0.2 \
