
mkdir ./model/api_seq_rec_ep30/outputs/results_beam_test
data_dir='../../../all_data/api_seq_data/codet5_data/codet5_format_data/'
python  run_gen2.py  --do_test --do_eval_bleu \
        --task refine --sub_task small --model_type codet5 --data_num -1    \
        --num_train_epochs  10  \
	--eval_steps 17000 \
        --warmup_steps 500 \
        --learning_rate 5e-5 --patience 3 --beam_size 5\
        --gradient_accumulation_steps 2 \
        --tokenizer_name=Salesforce/codet5-base  \
        --model_name_or_path=Salesforce/codet5-base \
        --data_dir  ${data_dir}    \
        --cache_path ./model/api_seq_rec_ep30/cache/  \
        --output_dir ./model/api_seq_rec_ep30/outputs/  \
        --summary_dir ./model/api_seq_rec_ep30/summary/   --save_last_checkpoints --always_save_model \
        --res_dir ./model/api_seq_rec_ep30/outputs/results_beam_test  \
        --res_fn  ./model/api_seq_rec_ep30/outputs/results_beam_test/summarize_codet5_base.txt  \
        --train_batch_size 96 --eval_batch_size 96 --max_source_length 256 --max_target_length 100  
