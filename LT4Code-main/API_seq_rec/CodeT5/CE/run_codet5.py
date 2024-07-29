import os
import logging
import argparse
import multiprocessing
import torch
import time
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer
from utils import load_and_cache_gen_data, get_filenames
from configs import add_args, set_dist, set_seed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if args.do_test:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        set_seed(args)
        set_dist(args)

        config = T5Config.from_pretrained(args.model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model.load_state_dict(torch.load(args.load_model_path))
        model.to(device)

        pool = multiprocessing.Pool(args.cpu_cont)
        args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
        eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test', only_src=True)

        # 制限して最初の10件分のみを使用する    コメントアウトしていればテストデータを全件読み込む
        # eval_examples = eval_examples[:10]
        # eval_data = TensorDataset(*[t[:10] for t in eval_data.tensors])

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("*** Start Prediction ***")
        predictions = []
        start_time = time.time()
        for batch_idx, batch in enumerate(eval_dataloader):
            batch_start_time = time.time()
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            with torch.no_grad():
                outputs = model.generate(input_ids=source_ids, attention_mask=source_mask, max_length=args.max_target_length, num_beams=args.beam_size)
                for output in outputs:
                    text = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    predictions.append(text)
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            total_batches = len(eval_dataloader)
            completed_batches = batch_idx + 1
            percent_complete = (completed_batches / total_batches) * 100
            logger.info(f"Batch {completed_batches}/{total_batches} completed ({percent_complete:.2f}%).")
            logger.info(f"Batch time: {batch_time:.2f} seconds. Total time elapsed: {time.time() - start_time:.2f} seconds.")
            logger.info(f"Predicted {len(predictions)} samples so far.")

        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "predictions.txt")
        with open(output_file, "w") as writer:
            for prediction in predictions:
                writer.write(prediction.strip() + "\n")
        logger.info(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
