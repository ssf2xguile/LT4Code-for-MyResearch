import json
import os
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
from transformers import RobertaTokenizerFast
from pandas import Series
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

weights = {
    '1': [1],
    '2': [1. / 2., 1. / 2.],
    '3': [1. / 3., 1. / 3., 1. / 3.],
    '4': [1. / 4., 1. / 4., 1. / 4., 1. / 4.]
}

def calculate_blue_samples(references, candidates, weight_index):
    if len(references) != len(candidates):
        raise ValueError('The number of sentences in both files do not match.')

    score = 0.
    chencherry = SmoothingFunction()
    for i in range(len(references)):
        score += sentence_bleu([
            references[i].strip().lower().split()],
            candidates[i].strip().lower().split(),
            weights=weights[weight_index],
            smoothing_function=chencherry.method0
        )
    score /= len(references)
    print("The bleu score BLEU-{} is: {}".format(weight_index, str(score)))
    return score

def parse_string_into_apis(str_):
    apis = []
    eles = str_.split('.')
    first_lib = eles[0]

    for i in range(1, len(eles) - 1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
            apis.append(first_lib.strip() + '.' + module_.strip())
            first_lib = library_
        except ValueError:
            try:
                module_, library_ = eles[i].strip().split(' ', 1)
                apis.append(first_lib.strip() + '.' + module_.strip())
                first_lib = library_
            except ValueError:
                module_ = eles[i].strip()
                library_ = ''
                apis.append(first_lib.strip() + '.' + module_.strip())
                first_lib = module_

    apis.append(first_lib.strip() + '.' + eles[-1].strip())
    return apis

def EM_samples(references, candidates):
    em = []
    for i in range(len(references)):
        if references[i].strip() == candidates[i].strip():
            em.append(1)
        else:
            em.append(0)
    return sum(em) / len(em)

def process_results_for_plot(ref_path, pred_path, vocab_tokens, freq_vocab, model_name=None):
    references = read_txt_file(pred_path)

    # Determine the file extension and read accordingly
    _, ext = os.path.splitext(ref_path)
    if ext == '.json':
        candidates = read_json_file(ref_path)
    elif ext == '.fixed':
        candidates = read_fixed_file(ref_path)
    elif ext == '.csv':
        candidates = read_csv_file(ref_path)
    else:
        raise ValueError("Unsupported file extension. Only .json and .fixed files are supported.")

    if model_name == 'codet5':
        candidates = [c.split('\t')[0].strip() for c in candidates]

    test_apis = []
    for d in tqdm(references):
        api_seqs = parse_string_into_apis(d)
        api_seqs = [a.lower() for a in api_seqs]
        test_apis.append(api_seqs)
    test_lt_score = []
    for l in test_apis:
        score_ = 0
        for api in l:
            try:
                score_ += 1 / (freq_vocab[api])
            except:
                score_ += 0
        score_ = score_ / math.sqrt(len(l))
        test_lt_score.append(score_)

    pos_list = [0.1 * i for i in range(1, 11)]
    threshold_list = [np.quantile(test_lt_score, pos) for pos in pos_list]
    metrics_by_ratios = []
    prior_test_thresh = min(test_lt_score)
    all_seq_d = []
    em_ = EM_samples(references, candidates)
    for i in range(len(threshold_list)):
        test_thresh = threshold_list[i]
        test_by_thresh_preds, test_by_thresh_golds = [], []
        for j in range(len(test_lt_score)):
            if test_lt_score[j] <= test_thresh and test_lt_score[j] > prior_test_thresh:
                test_by_thresh_preds.append(candidates[j])
                test_by_thresh_golds.append(references[j])
        all_seq_d.append(test_by_thresh_golds)
        if len(test_by_thresh_preds) > 0:
            em_ = EM_samples(test_by_thresh_golds, test_by_thresh_preds)
            metrics_by_ratios.append(em_)
        else:
            metrics_by_ratios.append(em_)
        prior_test_thresh = test_thresh
    return metrics_by_ratios

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return [' '.join(item['docstring_tokens']) for item in data]

def read_fixed_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df['target_api'].str.lower()  # Convert to lowercase, because ground trouth api sequence is presented in Capital character.

def Head_Tail_Sets_Results(ref_path, pred_path, vocab_tokens, freq_vocab, threshold, model_name=None):
    references = read_txt_file(pred_path)

    # Determine the file extension and read accordingly
    _, ext = os.path.splitext(ref_path)
    if ext == '.json':
        candidates = read_json_file(ref_path)
    elif ext == '.fixed':
        candidates = read_fixed_file(ref_path)
    elif ext == '.csv':
        candidates = read_csv_file(ref_path)
    else:
        raise ValueError("Unsupported file extension. Only .json and .fixed files are supported.")

    if model_name == 'codet5':
        candidates = [c.split('\t')[0].strip() for c in candidates]

    test_apis = []
    for d in tqdm(references):
        api_seqs = parse_string_into_apis(d)
        api_seqs = [a.lower() for a in api_seqs]
        test_apis.append(api_seqs)
    test_lt_score = []
    for l in test_apis:
        score_ = 0
        for api in l:
            try:
                score_ += 1 / (freq_vocab[api])
            except:
                continue
        score_ = score_ / math.sqrt(len(l))
        test_lt_score.append(score_)

    head_preds, head_labels, tail_preds, tail_labels = [], [], [], []

    for ij in range(len(test_lt_score)):
        if test_lt_score[ij] <= threshold:
            head_preds.append(candidates[ij])
            head_labels.append(references[ij])
        else:
            tail_preds.append(candidates[ij])
            tail_labels.append(references[ij])

    print('ALL:', len(references))
    em_ = EM_samples(references, candidates)
    print('EM:', em_)
    print('HEAD:', len(head_preds))
    em_ = EM_samples(head_labels, head_preds)
    print('EM:', em_)
    print('TAIL:', len(tail_preds))
    em_ = EM_samples(tail_labels, tail_preds)
    print('EM:', em_)

def main():
    splits = ['train', 'test', 'validate']
    data_dir = '../all_data/api_seq_data/mularec_data/'
    all_data = []
    for split in splits:
        file_name = data_dir + split + '_3_lines.csv'
        df = pd.read_csv(file_name)
        all_data.append(df)

    all_data = pd.concat(all_data)
    all_input_doc = list(all_data['annotation'])
    all_input_code = list(all_data['source_code'])
    all_inputs = [all_input_doc[i] + all_input_code[i] for i in range(len(all_input_code))]
    all_data = list(all_data['target_api'])

    all_api_data = []
    all_apis = []
    for d in tqdm(all_data):
        api_seqs = parse_string_into_apis(d)
        api_seqs = [a.lower() for a in api_seqs]
        all_api_data.extend(api_seqs)
        all_apis.append(api_seqs)
    all_data = all_api_data

    api_rec_ids = list(range(len(all_data)))
    api_rec_labels = all_data
    api_rec_categories = ['api_sequence_rec'] * len(all_data)
    vocab = Counter(api_rec_labels)
    vocab_tokens = [i[0] for i in vocab.most_common(len(vocab))]
    vocab_samples = [i[1] for i in vocab.most_common(len(vocab))]
    total_sample = sum(vocab_samples)
    freq_vocab = {vocab_tokens[i]: vocab_samples[i] / total_sample for i in range(len(vocab_tokens))}

    head_classes, tail_classes = [], []
    cumulative = 0
    for i in range(len(vocab_samples)):
        cumulative += vocab_samples[i] / total_sample
        if cumulative <= 0.5:
            head_classes.append(vocab_tokens[i])
        else:
            tail_classes.append(vocab_tokens[i])

    print(np.sum(vocab_samples[0:len(head_classes)]) / total_sample)
    print(vocab_samples[len(head_classes) + 1])

    all_apis_each_sample_scores = []
    for l in all_apis:
        score_ = 0
        for api in l:
            score_ += 1 / (freq_vocab[api])
        score_ = score_ / math.sqrt(len(l))
        all_apis_each_sample_scores.append(score_)
    print(len(all_apis_each_sample_scores), all_apis_each_sample_scores[0:10])
    threshold = np.quantile(all_apis_each_sample_scores, .50)

    Head_Tail_Sets_Results(
        "../API_seq_rec/Integration/data/test.json",
        "../API_seq_rec/Integration/data/CodeBERT_predictions.txt",
        vocab_tokens, freq_vocab, threshold)

    Head_Tail_Sets_Results(
        "../API_seq_rec/Integration/data/test.buggy-fixed.fixed",
        "../API_seq_rec/Integration/data/CodeT5_predictions.txt",
        vocab_tokens, freq_vocab, threshold, 'codet5')
    
    Head_Tail_Sets_Results(
        "../API_seq_rec/Integration/data/test_3_lines.csv",
        "../API_seq_rec/Integration/data/MulaRec_predictions.txt",
        vocab_tokens, freq_vocab, threshold)

    metrics_by_ratios1 = process_results_for_plot(
        "../API_seq_rec/Integration/data/test.json",
        "../API_seq_rec/Integration/data/CodeBERT_predictions.txt",
        vocab_tokens, freq_vocab)

    metrics_by_ratios2 = process_results_for_plot(
        "../API_seq_rec/Integration/data/test.buggy-fixed.fixed",
        "../API_seq_rec/Integration/data/CodeT5_predictions.txt",
        vocab_tokens, freq_vocab, 'codet5')
    
    metrics_by_ratios3 = process_results_for_plot(
        "../API_seq_rec/Integration/data/test_3_lines.csv",
        "../API_seq_rec/Integration/data/MulaRec_predictions.txt",
        vocab_tokens, freq_vocab)

    scale = 0.6
    ids = list(range(len(metrics_by_ratios1)))
    series = pd.DataFrame({'MulaRec':metrics_by_ratios3, 'CodeT5':metrics_by_ratios2, 'CodeBERT': metrics_by_ratios1, 'x':list(range(1,11,1))})
    rolling = series.rolling(window=1)
    rolling_mean = rolling.mean()
    #print(rolling_mean.head(10))
    rolling_mean.plot(x='x',color=['orangered','lightgreen', 'violet'], kind='line', style=['-','-.','--'], linewidth=2, alpha=1, figsize=(9.6*scale, 7.2*scale), fontsize=14)
    plt.xlabel('Sorted Groups ID', fontsize=14)
    plt.ylabel('EM', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.savefig('../API_seq_rec/Integration/data/Exact_Match_Rate.png', format='png', dpi=200)
if __name__ == "__main__":
    main()
