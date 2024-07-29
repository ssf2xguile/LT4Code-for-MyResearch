mkdir model
mkdir ./model/api_seq_rec_fl/
mkdir ./model/api_seq_rec_fl/cache/
mkdir ./model/api_seq_rec_fl/outputs/
mkdir ./model/api_seq_rec_fl/summary/
mkdir ./model/api_seq_rec_fl/outputs/results
data_dir='../../../all_data/api_seq_data/codet5_data/codet5_format_data/'

python  LT_run_gen.py  --do_train --do_eval  \
        --LT_solution 'fl' \
        --task refine --sub_task small --model_type codet5 --data_num -1    \
	      --eval_steps 17000 \
	      --num_train_epochs  10  \
        --warmup_steps 500 \
        --learning_rate 5e-5 --patience 3 --beam_size 10 \
        --gradient_accumulation_steps 2 \
        --tokenizer_name=Salesforce/codet5-base  \
        --model_name_or_path=Salesforce/codet5-base \
        --data_dir  ${data_dir}    \
        --cache_path ./model/api_seq_rec_fl/cache/  \
        --output_dir ./model/api_seq_rec_fl/outputs/  \
        --summary_dir ./model/api_seq_rec_fl/summary/   --save_last_checkpoints --always_save_model \
        --res_dir ./model/api_seq_rec_fl/outputs/results \
        --res_fn  ./model/api_seq_rec_fl/outputs/results/summarize_codet5_base.txt  \
	      --sample_per_class_file  ${data_dir}refine/small/token_freq_file.pkl \
        --train_batch_size 16 --eval_batch_size 16 --max_source_length 256 --max_target_length 100 