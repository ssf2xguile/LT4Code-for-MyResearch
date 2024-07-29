"""CodeBERTをお試しで動かすプログラム
テストデータ10件を入力し、10件分の出力をテキストファイル(LT4Code-main/API_seq_rec/CodeBERT/CE/model/codebert_epoch30_ce/predicts/predictions.txt)に書き込む。
カレントディレクトリ: LT4Code-main/API_seq_rec/CodeBERT/CE
使い方: python run_codebert.py --do_test
"""
from __future__ import absolute_import
import time
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # if idx > 10:  テストデータ全件を生成する場合はコメントアウトする
                # break
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, example_id, source_ids, source_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_mask = source_mask

def convert_examples_to_features(examples, tokenizer, args):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))
            logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
            logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
            logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                source_mask,
            )
        )
    return features

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default="model/codebert_epoch30_ce/predicts/", type=str,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--load_model_path", default="model/codebert_epoch30_ce/checkpoint-best-ppl/pytorch_model.bin", type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--test_filename", default="../../../all_data/api_seq_data/codebert_data/test.json", type=str,
                        help="The test filename. Should contain the .json files for this task.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--max_source_length", default=256, type=int,
                        help="The maximum total source sequence length after tokenization.")
    parser.add_argument("--max_target_length", default=100, type=int,
                        help="The maximum total target sequence length after tokenization.")
    parser.add_argument("--beam_size", default=5, type=int,
                        help="Beam size for beam search")
    parser.add_argument("--eval_batch_size", default=96, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Set seed
    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    
    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    model.load_state_dict(torch.load(args.load_model_path))
    model.to(device)

    # Load and preprocess test examples
    eval_examples = read_examples(args.test_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args)
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids, all_source_mask)

    # Create test dataloader
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("*** Start Prediction ***")
    # Perform test
    predictions = []
    start_time = time.time()
    for batch_idx, batch in enumerate(eval_dataloader):
        batch_start_time = time.time()
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            outputs = model(source_ids=source_ids, source_mask=source_mask) 
            for output in outputs:
                candit = []
                for single_pred in output:
                    t = single_pred.cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    candit.append(text)
                candit_text = '\t'.join(candit)
                predictions.append(candit_text)
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        total_batches = len(eval_dataloader)
        completed_batches = batch_idx + 1
        percent_complete = (completed_batches / total_batches) * 100
        logger.info(f"Batch {completed_batches}/{total_batches} completed ({percent_complete:.2f}%).")
        logger.info(f"Batch time: {batch_time:.2f} seconds. Total time elapsed: {time.time() - start_time:.2f} seconds.")
        logger.info(f"Predicted {len(predictions)} samples so far.")


    # Save predictions
    with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
        for prediction in predictions:
            f.write(prediction.strip() + "\n")
            
if __name__ == "__main__":
    main()