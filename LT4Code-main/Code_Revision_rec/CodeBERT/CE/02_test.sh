
data_dir='../../../all_data/code_review_data/codebert_data'
CUDA_VISIBLE_DEVICES=7 python run2_12_2.py --do_test --model_type roberta      \
      --load_model_path   'model/code_doc2code/checkpoint-best-bleu/pytorch_model.bin' \
      --output_dir model/code_doc2code/out_beam5_hyp5/    \
      --model_name_or_path microsoft/codebert-base   \
      --test_filename ${data_dir}/test.json \
      --max_source_length 300 --max_target_length 200  \
      --beam_size 5   \
      --train_batch_size 8 --eval_batch_size 32        \
      --learning_rate 5e-5   \
      --train_steps 160000  --eval_steps 7500 
