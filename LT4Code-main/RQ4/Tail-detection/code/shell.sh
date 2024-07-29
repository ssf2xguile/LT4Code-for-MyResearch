#!/bin/bash

python run_lightgbm_to_find_bestLR.py \
    --train_data_file=../../../all_data/RQ4_data/api_data/train.jsonl \
    --eval_data_file=../../../all_data/RQ4_data/api_data/valid.jsonl \
    --test_data_file=../../../all_data/RQ4_data/api_data/test.jsonl 2>&1 | tee train_api_lightgbm.log