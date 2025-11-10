#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../main_github.py \
--data_name "mimic_cxr" \
--version "best" \
--task "pretraining" \
--phase "inference" \
--ann_path "/home/miao/data/dataset/MIMIC-CXR/five_work_mimic_cxr_annotation_v2.json" \
--view_position_dict "/home/miao/data/dataset/MIMIC-CXR/view-positions-dict-mimic.json" \
--images_dir "/home/miao/data/dataset/MIMIC-CXR/files/" \
--max_length 100 \
--encoder_max_length 300 \
--num_workers 6 \
--is_save_checkpoint "yes" \
--ckpt_zoo_dir "/home/miao/data/dataset/checkpoints" \
--test_ckpt_path "../checkpoints/mimic-cxr/v0207-align_2025_02_07_22-best/checkpoint/best_model.ckpt" \
--temporal_fusion_num_blocks 3 \
--perceiver_num_blocks 3 \
--num_latents 128 \
--patience 10 \
--pt_lr 5.0e-5 \
--epochs 10 \
--batch_size 32