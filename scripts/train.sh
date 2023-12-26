#!/bin/bash
# max_seq_lenght: 512 for bge-large-en-v1.5, 384 for all_mpnet-base-v2

python -u ../finetune/train.py 2>&1 \
    --prefix_path ../data/dataset \
    --model_name bge-large-en-v1.5 \
    --pretrained_model_path ../model/bge-large-en-v1.5 \
    --train_batch_size 8 \
    --max_seq_length 512 \
    --num_epochs 15 \
    --dev_data_path ../data/row-dataset/val_2021.qrels.pass.final.txt \
    --model_save_path ../model/bge-large-en-v1.5-finetuned \
    | tee ../log/train.log 