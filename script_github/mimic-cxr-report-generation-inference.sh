#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../main_github.py \
--data_name "mimic_cxr" \
--version "best" \
--task "report-generation-gpt2" \
--phase "inference" \
--ann_path "/MIMIC-CXR/priorrg_mimic_cxr_annotation.json" \
--view_position_dict "/MIMIC-CXR/view-positions-dict-mimic.json" \
--images_dir "/MIMIC-CXR/files/" \
--max_length 100 \
--encoder_max_length 300 \
--num_workers 6 \
--is_save_checkpoint "no" \
--test_ckpt_path "../checkpoints/mimic-cxr/v0307-all-have_2025_03_18_09-best/checkpoint/best_model.ckpt" \
--ckpt_zoo_dir "/dataset/checkpoints" \
--temporal_fusion_num_blocks 3 \
--perceiver_num_blocks 3 \
--num_latents 128 \
--patience 5 \
--pt_lr 5.0e-6 \
--ft_lr 5.0e-5 \
--monitor_metric "RCB" \
--epochs 50 \
--batch_size 16
