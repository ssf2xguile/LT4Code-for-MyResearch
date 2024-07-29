#!/bin/bash

python run_xgboost.py \
    --output_dir=./saved_models_api_xgboost \
    --train_data_file=../../../all_data/RQ4_data/api_data/train.jsonl \
    --eval_data_file=../../../all_data/RQ4_data/api_data/valid.jsonl \
    --test_data_file=../../../all_data/RQ4_data/api_data/test.jsonl \
    --model_filename=xgboost.pkl 2>&1 | tee train_api_xgboost.log