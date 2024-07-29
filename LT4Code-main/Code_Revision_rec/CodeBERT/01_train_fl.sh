data_dir='../../all_data/code_review_data/codebert_data'
mkdir model/code_doc2code_LT_fl/
CUDA_VISIBLE_DEVICES=7 python LT_run12.py --do_train --do_eval --model_type roberta      \
      --LT_solution  'fl' \
      --sample_per_class_file  '../data/codebert_token_freq_file.pkl' \
      --model_name_or_path microsoft/codebert-base   \
      --train_filename ${data_dir}/train.json       \
      --dev_filename   ${data_dir}/val.json     \
      --output_dir model/code_doc2code_LT_fl/   \
      --max_source_length 300 --max_target_length 200  \
      --beam_size 5   \
      --train_batch_size 8 --eval_batch_size 8        \
      --learning_rate 5e-5   \
      --train_steps 160000  --eval_steps 7500  
