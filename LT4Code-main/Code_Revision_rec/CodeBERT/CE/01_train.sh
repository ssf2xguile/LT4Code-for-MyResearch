data_dir='../../../all_data/code_review_data/codebert_data'
mkdir model/code_doc2code/
CUDA_VISIBLE_DEVICES=4,5 python run_12.py --do_train --do_eval --model_type roberta      \
     	--model_name_or_path microsoft/codebert-base   \
 	--train_filename ${data_dir}/train.json       \
     	--dev_filename   ${data_dir}/val.json     \
     	--output_dir model/code_doc2code/   \
 	--max_source_length 300 --max_target_length 200  \
 	--beam_size 5   \
 	--train_batch_size 8 --eval_batch_size 8        \
        --learning_rate 5e-5   \
        --train_steps 80000  --eval_steps 7500
