
CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_type 2 \
	--load_model_path dual-mularec-fl-epoch30/checkpoint-best-bleu/pytorch_model.bin \
	--test_filename  ../../../all_data/api_seq_data/mularec_data/test_3_lines.csv \
	--output_dir  dual-mularec-fl-epoch30/res-1 \
	--max_length 256
