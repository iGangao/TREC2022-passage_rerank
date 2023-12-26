#!/bin/bash
MODEL_PATH=./model/bge-reranker-large
OUTPUT_FILE=./result/re-ranking_result-bge-re.txt
TEST_QRELS_FILE=./data/row-dataset/test_2022.qrels.pass.withDupes.txt


# python -u ./process_data.py 2>&1 | tee ./log/process_data.log

python -u ./rerank.py 2>&1 \
    --model_path ${MODEL_PATH} \
    --output_file ${OUTPUT_FILE} \
    --corpus_file ./data/dataset/corpus_pid2passage.json \
    --id2query_file ./data/dataset/test_qid2query.json \
    --id2pids_file ./data/dataset/test_qid2pids.json \
    | tee ./log/re_ranking.log 

./trec_eval-9.0.7/trec_eval -m ndcg_cut $TEST_QRELS_FILE $OUTPUT_FILE
