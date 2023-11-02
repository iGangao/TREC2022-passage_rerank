#!/bin/bash
TEST_QRELS_FILE=../data/row-dataset/test_2022.qrels.pass.withDupes.txt
VAL_QRELS_FILE=../data/row-dataset/val_2021.qrels.pass.final.txt
RESULT_FILE=../result/test_ranking_result.txt
../trec_eval-9.0.7/trec_eval -m ndcg_cut $TEST_QRELS_FILE $RESULT_FILE
