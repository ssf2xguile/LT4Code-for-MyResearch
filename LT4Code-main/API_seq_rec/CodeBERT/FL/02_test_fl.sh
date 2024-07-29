data_dir='../../../all_data/api_seq_data/codebert_data/'
mkdir model/codebert_epoch30_fl/predicts/
python run2_12.py --do_test  --model_type roberta      \
     	--model_name_or_path microsoft/codebert-base   \
 	--test_filename  ${data_dir}test.json       \
     	--load_model_path  model/codebert_epoch30_fl/checkpoint-best-ppl/pytorch_model.bin  \
     	--output_dir model/codebert_epoch30_fl/predicts/   \
 	--max_source_length 256 --max_target_length 100  --beam_size 10   --train_batch_size 96  --eval_batch_size 96        \
        --learning_rate 5e-5  






