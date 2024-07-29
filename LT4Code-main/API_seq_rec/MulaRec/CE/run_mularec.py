import os
import argparse
import logging
from models.dual_model import MulaRecDual
import torch
import time
from utils import read_examples, convert_examples_to_features
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=int, required=True, help="model type")
    parser.add_argument("--max_length", type=int, required=True, help="max length")
    parser.add_argument("--load_model_path", type=str, required=True, help="the fine-tuned model path")
    parser.add_argument("--test_filename", type=str, required=True, help="the test file name")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)    
    set_logger(os.path.join(args.output_dir, 'eval.log'))
    logging.info(args)
    
    if args.model_type == 2:
        logging.info('***' * 10)
        logging.info('dual model')
        
        model = MulaRecDual(
            codebert_path='microsoft/codebert-base', 
            decoder_layers=6, 
            fix_encoder=False, 
            beam_size=10,
            max_source_length=args.max_length, 
            max_target_length=args.max_length, 
            load_model_path=args.load_model_path,
            l2_norm=True,
            fusion=True
        )

    # test model
    eval_examples = read_examples(args.test_filename)
    #eval_examples = eval_examples[:10]  # 最初の10件を使用
    eval_features = convert_examples_to_features(eval_examples, model.tokenizer, args.max_length, args.max_length, stage='test')

    all_src_ant_ids = torch.tensor([f.src_ant_ids for f in eval_features], dtype=torch.long)
    all_src_ant_mask = torch.tensor([f.src_ant_mask for f in eval_features], dtype=torch.long)

    all_src_code_ids = torch.tensor([f.src_code_ids for f in eval_features], dtype=torch.long)
    all_src_code_mask = torch.tensor([f.src_code_mask for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_src_ant_ids, all_src_ant_mask, all_src_code_ids, all_src_code_mask)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=8)

    model.model.eval()
    logger.info("*** Start Prediction ***")
    predictions = []
    start_time = time.time()
    for batch_idx, batch in enumerate(eval_dataloader):
        batch_start_time = time.time()
        batch = tuple(t.to(model.device) for t in batch)
        src_ant_ids, src_ant_mask, src_code_ids, src_code_mask = batch

        with torch.no_grad():
            preds = model.model(
                src_ant_ids=src_ant_ids, 
                src_ant_mask=src_ant_mask, 
                src_code_ids=src_code_ids, 
                src_code_mask=src_code_mask
            )
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = model.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                predictions.append(text)
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        total_batches = len(eval_dataloader)
        completed_batches = batch_idx + 1
        percent_complete = (completed_batches / total_batches) * 100
        logger.info(f"Batch {completed_batches}/{total_batches} completed ({percent_complete:.2f}%).")
        logger.info(f"Batch time: {batch_time:.2f} seconds. Total time elapsed: {time.time() - start_time:.2f} seconds.")
        logger.info(f"Predicted {len(predictions)} samples so far.")

    output_file = os.path.join(args.output_dir, "predictions.txt")
    with open(output_file, "w") as writer:
        for prediction in predictions:
            writer.write(prediction.strip() + "\n")
    logging.info(f"Predictions saved to {output_file}")

if __name__ == '__main__':     
    main()
    