mkdir model
mkdir ./model/code_doc2code/
mkdir ./model/code_doc2code/cache/
mkdir ./model/code_doc2code/outputs/
mkdir ./model/code_doc2code/summary/
mkdir ./model/code_doc2code/outputs/results
data_dir='../../../all_data/code_review_data/codet5_data/codet5_format_data/'

CUDA_VISIBLE_DEVICES=5 python  run_gen.py  --do_train --do_eval --do_eval_bleu  \
        --task refine --sub_task small --model_type codet5 --data_num -1    \
        --eval_steps 30000\
        --num_train_epochs  10  \
        --warmup_steps 500 \
        --learning_rate 5e-5 --patience 3 --beam_size 5\
        --gradient_accumulation_steps 2 \
        --tokenizer_name=Salesforce/codet5-base  \
        --model_name_or_path=Salesforce/codet5-base \
        --data_dir  ${data_dir}    \
        --cache_path ./model/code_doc2code/cache/  \
        --output_dir ./model/code_doc2code/outputs/  \
        --summary_dir ./model/code_doc2code/summary/   --save_last_checkpoints --always_save_model \
        --res_dir ./model/code_doc2code/outputs/results \
        --res_fn  ./model/code_doc2code/outputs/results/summarize_codet5_base.txt  \
        --train_batch_size 4 --eval_batch_size 4 --max_source_length 300 --max_target_length 250
