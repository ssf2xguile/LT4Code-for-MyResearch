data_dir='../../../all_data/api_seq_data/codebert_data/'
mkdir model
mkdir model/codebert_epoch30_ce/
python run_12.py --do_train --do_eval --model_type roberta      \
     	--model_name_or_path microsoft/codebert-base   \
 	--train_filename ${data_dir}train.json       \
     	--dev_filename  ${data_dir}valid.json      \
     	--output_dir model/codebert_epoch30_ce/   \
 	--max_source_length 256 --max_target_length 100  --beam_size 10  --train_batch_size 32  --eval_batch_size 16        \
        --learning_rate 5e-5   --train_steps 96000  --eval_steps 3100 
