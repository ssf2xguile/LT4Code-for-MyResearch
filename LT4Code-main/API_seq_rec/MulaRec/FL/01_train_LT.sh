CUDA_VISIBLE_DEVICES=1 python LT_dual_run.py --max_length 256 \
          --batch_size 32 --epoch 30 \
          --fuse True --norm True \
          --LT_solution fl \
          --output_dir 'dual-mularec-fl-epoch30' \
	  --data_dir '../../../all_data/api_seq_data/mularec_data/'
