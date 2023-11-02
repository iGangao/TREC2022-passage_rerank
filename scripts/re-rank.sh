#!/bin/bash
MODEL_PATH=../model/all-mpnet-base-v2
OUTPUT_FILE=../result/re-ranking_result.txt
python -u ../rerank.py 2>&1 \
    --model_path ${MODEL_PATH} \
    --output_file ${OUTPUT_FILE} \
    | tee ../log/re_ranking.log 