#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../main_github.py \
--data_name "mimic_cxr" \
--version "best" \
--task "pretraining" \
--phase "finetune" \
--ann_path "priorrg_mimic_cxr_annotation.json" \
--view_position_dict "view-positions-dict-mimic.json" \
--images_dir "dataset/MIMIC-CXR/files/" \
--max_length 100 \
--encoder_max_length 300 \
--num_workers 6 \
--is_save_checkpoint "yes" \
--ckpt_zoo_dir "/dataset/checkpoints" \
--temporal_fusion_num_blocks 3 \
--perceiver_num_blocks 3 \
--num_latents 128 \
--patience 10 \
--pt_lr 5.0e-5 \
--epochs 10 \
--batch_size 32
