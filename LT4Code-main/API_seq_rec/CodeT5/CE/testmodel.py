"""自作ファイル"""
import os
import argparse
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from utils import load_and_cache_gen_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="codet5", type=str)
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base", type=str)
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-base", type=str)
    parser.add_argument("--output_dir", default="./model/api_seq_rec/outputs/", type=str)
    parser.add_argument("--data_dir", default="../../../all_data/api_seq_data/codet5_data/codet5_format_data/", type=str)
    parser.add_argument("--test_filename", default="test.json", type=str)
    parser.add_argument("--max_source_length", default=256, type=int)
    parser.add_argument("--max_target_length", default=100, type=int)
    parser.add_argument("--beam_size", default=5, type=int)
    args = parser.parse_args()

    # モデルとトークナイザーのロード
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(os.path.join(args.output_dir, "checkpoint-last/pytorch_model.bin"))
    model.to('cuda')

    # テストデータのロード
    eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, None, tokenizer, 'test', only_src=True, is_sample=False)

    # 推論
    model.eval()
    with torch.no_grad():
        for index, example in enumerate(eval_examples):
            if index >= 10:  # 最初の10件だけ処理
                break

            source_ids = tokenizer.encode(example.source, max_length=args.max_source_length, padding='max_length', truncation=True, return_tensors="pt").to('cuda')
            outputs = model.generate(source_ids, 
                                     max_length=args.max_target_length, 
                                     num_beams=args.beam_size,
                                     num_return_sequences=1,
                                     early_stopping=True)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(f"Example {index + 1}:")
            print("Source:", example.source)
            print("Generated:", decoded)
            print()

if __name__ == "__main__":
    main()