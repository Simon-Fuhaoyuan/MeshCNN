#!/usr/bin/env bash

## run the training
python train.py \
--dataroot /disk5/data/MeshCNN/datasets/debug \
--name debug \
--arch meshunet_recon \
--dataset_mode reconstruction \
--recon_subroot alien \
--ncf 64 128 256 256 \
--pool_res 600 450 300 \
--resblocks 3 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
