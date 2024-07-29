
import json
from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import json
import json,os, random
import pandas as pd
from collections import Counter
from tqdm import  tqdm
import numpy as np
import numpy as np
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

###Review Code Edit Data
from tqdm import tqdm
import difflib
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")


def get_token_level_edit_pair(src, tgt):
    all_edits = []
    det = src
    add = tgt
    det = tokenizer.tokenize(det)
    add = tokenizer.tokenize(add)

    s = difflib.SequenceMatcher(None, det, add)
    longpath = s.find_longest_match(0, len(det), 0, len(add))
    matching = s.get_matching_blocks()

    for tag, i1, i2, j1, j2 in s.get_opcodes():

        if tag == 'delete':
            for src_token in det[i1:i2]:
                all_edits.append((src_token, ''))
        elif tag == 'insert':
            for tgt_token in add[j1:j2]:
                all_edits.append(('', tgt_token))
        elif tag == 'replace':
            for src_token in det[i1:i2]:
                all_edits.append((src_token, ''))
            for tgt_token in add[j1:j2]:
                all_edits.append(('', tgt_token))
    return all_edits


splits = ['train', 'test', 'valid']
data_dir = '../all_data/code_review_data/codet5_data//codet5_format_data/refine/small/'
all_sources, all_golds = [], []
test_sources, test_golds = [], []
for split in splits:
    file_name = data_dir.rsplit('/', 2)[0] + '/raw/' + 'src-' + split + '.txt'  ##Code Only
    with open(file_name, 'r') as f:
        sources = f.readlines()
        all_sources.extend(sources)
        if split == 'test':
            test_sources.extend(sources)
    file_name = data_dir + split + '.buggy-fixed.fixed'
    with open(file_name, 'r') as f:
        golds = f.readlines()
        all_golds.extend(golds)
        if split == 'test':
            test_golds.extend(golds)

all_edits_all_data = []
all_edits = []
for i in tqdm(range(len(all_sources))):
    edit_ = get_token_level_edit_pair(all_sources[i], all_golds[i])
    all_edits.append(edit_)
    all_edits_all_data.extend(edit_)
all_data = all_edits_all_data


vocab = Counter(all_data)
vocab_tokens = [i[0] for i in vocab.most_common(len(vocab))]
vocab_samples = [i[1] for i in vocab.most_common(len(vocab))]
total_smaple = sum(vocab_samples)
freq_vocab = {}
for i in range(len(vocab_tokens)):
    freq_vocab[vocab_tokens[i]] = vocab_samples[i] / total_smaple  ##原始的freq

all_edits_each_smaple_scores = []
for l in all_edits:
    score_ = 0
    for edit in l:
        score_ += 1 / (freq_vocab[edit])
    score_ = score_ / math.sqrt(len(l))
    all_edits_each_smaple_scores.append(score_)

threshold = np.quantile(all_edits_each_smaple_scores, .50)

head_classes, tail_classes = [], []
cumulative = 0
for i in range(len(vocab_samples)):
    cumulative += vocab_samples[i] / total_smaple
    if cumulative <= 0.5:
        head_classes.append(vocab_tokens[i])
    else:
        tail_classes.append(vocab_tokens[i])
#print('head class:', head_classes[0:10], len(head_classes))
#print('tail class:', tail_classes[0:10], len(tail_classes))
print(np.sum(vocab_samples[0:len(head_classes)]) / total_smaple)
print(vocab_samples[len(head_classes) + 1])

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
import math

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
    #print("The bleu score BLEU-{} is: {}".format(weight_index, str(score)))
    return score


def exact_match(preds, goldens, top=1):
    correct_ = 0
    if top == 1:
        for i in range(len(preds)):
            prediction = preds[i].strip()  ## only the top 1 recommendation
            prediction = ' '.join(prediction.split())
            gold = ' '.join(goldens[i].strip().split())
            if prediction.strip() == gold.strip():
                correct_ += 1
        #print('Eaxct Match: ', correct_ / len(preds))
    else:
        for i in range(len(preds)):
            if goldens[i].strip() in [p.replace('\t', ' ').strip() for p in preds[i]]:
                correct_ += 1
        #print('Eaxct Match: ', correct_ / len(preds))

    return correct_ / len(preds)


def process_results_for_plot(result_dir, vocab_tokens, freq_vocab, model_name):
    f1 = open(result_dir + 'test.output', 'r')
    predictions = f1.readlines()

    return_num = 1
    if model_name == 'codet5':
        predictions = [p.split('\t')[0] for p in predictions]

    elif model_name == 't5_review':
        new_predictions = []
        for i in range(len(predictions)):
            if i % 5 == 0:
                new_predictions.append(predictions[i])
        predictions = new_predictions

    elif model_name == 'codebert':
        predictions = [p.split('\t')[1] for p in predictions]

    f2 = open(result_dir + 'test.gold', 'r')
    golds = f2.readlines()
    
    if model_name == 'codebert':
        golds = [p.split('\t')[1] for p in golds]

    f3 = open(result_dir + 'source.txt', 'r')
    sources = f3.readlines()
    
    test_edits_all_data = []
    for i in tqdm(range(len(sources))):
        test_edits_all_data.append(get_token_level_edit_pair(sources[i], golds[i]))

    test_edits_each_smaple_scores = []
    for l in test_edits_all_data:
        score_ = 0
        for edit in l:
            if edit not in freq_vocab:
                continue
            else:
                score_ += 1 / (freq_vocab[edit])
        score_ = score_ / math.sqrt(len(l))

        test_edits_each_smaple_scores.append(score_)

    pos_list = [0.1 * i for i in range(1, 11)]

    threshold_list = [np.quantile(test_edits_each_smaple_scores, pos) for pos in pos_list]

    metrics_by_ratios = []
    prior_test_thresh = min(test_edits_each_smaple_scores)
    all_seq_d = []
    em_ = exact_match(predictions, golds, top=1)
    for i in range(len(threshold_list)):
        test_thresh = threshold_list[i]
        test_by_thresh_preds, test_by_thresh_golds = [], []
        for j in range(len(test_edits_each_smaple_scores)):
            if test_edits_each_smaple_scores[j] < test_thresh and test_edits_each_smaple_scores[j] >= prior_test_thresh:
                test_by_thresh_preds.append(predictions[j])
                test_by_thresh_golds.append(golds[j])
        all_seq_d.append(test_by_thresh_golds)
        
        if len(test_by_thresh_preds) > 0:
            em_ = exact_match(test_by_thresh_preds, test_by_thresh_golds, top=1)
            metrics_by_ratios.append(em_)
        prior_test_thresh = test_thresh


    return metrics_by_ratios


def Head_Tail_Sets_Results(result_dir, vocab_tokens, freq_vocab, threshold, model_name):
    f1 = open(result_dir + 'test.output', 'r')
    predictions = f1.readlines()

    if model_name == 'codet5':
        predictions = [p.split('\t')[0] for p in predictions]


    elif model_name == 't5_review':
        new_predictions = []
        for i in range(len(predictions)):
            if i % 5 == 0:
                new_predictions.append(predictions[i])
        predictions = new_predictions


    elif model_name == 'codebert':
        predictions = [p.split('\t')[1] for p in predictions]

    f2 = open(result_dir + 'test.gold', 'r')
    golds = f2.readlines()

    if model_name == 'codebert':
        golds = [p.split('\t')[1] for p in golds]

    f3 = open(result_dir + 'source.txt', 'r')
    sources = f3.readlines()
    

    test_edits_all_data_ = []
    for i in tqdm(range(len(sources))):
        test_edits_all_data_.append(get_token_level_edit_pair(sources[i], golds[i]))

    test_edits_each_smaple_scores_ = []
    for l in test_edits_all_data_:
        score_ = 0
        for edit in l:
            if edit not in freq_vocab:
                continue
            else:
                score_ += 1 / (freq_vocab[edit])
        score_ = score_ / math.sqrt(len(l))

        test_edits_each_smaple_scores_.append(score_)

    head_preds, head_labels, tail_preds, tail_labels = [], [], [], []
    for ij in range(len(test_edits_each_smaple_scores_)):
        if test_edits_each_smaple_scores_[ij] >= threshold:
            tail_preds.append(predictions[ij])
            tail_labels.append(golds[ij])
        else:
            head_preds.append(predictions[ij])
            head_labels.append(golds[ij])
    print('ALL:', len(predictions))
    print('EM: ', exact_match(predictions, golds, top=1))
    #calculate_blue_samples(golds, predictions, '4')

    print('HEAD:', len(head_preds))
    print('EM: ',exact_match(head_preds, head_labels, top=1))
    #calculate_blue_samples(head_labels, head_preds, '4')

    print('TAIL:', len(tail_preds))
    print('EM: ',exact_match(tail_preds, tail_labels, top=1))
    #calculate_blue_samples(tail_labels, tail_preds, '4')


"""
metrics_by_ratios1 = process_results_for_plot(
    '../generated_predictions/revision_rec/T5_review/CE/', vocab_tokens, freq_vocab, 't5_review')
metrics_by_ratios2 = process_results_for_plot('../generated_predictions/revision_rec/CodeT5/CE/',
                                              vocab_tokens, freq_vocab, 'codet5')
metrics_by_ratios3 = process_results_for_plot(
    '../generated_predictions/revision_rec/CodeBERT/CE/', vocab_tokens, freq_vocab, 'codebert')

from pandas import Series
from matplotlib.pyplot import MultipleLocator

scale = 0.6
ids = list(range(len(metrics_by_ratios1)))
series = pd.DataFrame({'T5-review': metrics_by_ratios1, 'CodeT5': metrics_by_ratios2, 'CodeBERT': metrics_by_ratios3,
                       'x': list(range(1, 11, 1))})
rolling = series.rolling(window=1)
rolling_mean = rolling.mean()
rolling_mean.plot(x='x', color=['orangered', 'lightgreen', 'violet'], kind='line', style=['-', '-.', '--'], linewidth=2,
                  alpha=1, figsize=(9.6 * scale, 7.2 * scale), fontsize=14)
plt.xlabel('Sorted Groups ID', fontsize=14)
plt.ylabel('EM', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
"""



Head_Tail_Sets_Results('../generated_predictions/revision_rec/CodeT5/CE/', vocab_tokens, freq_vocab, threshold, 'codet5')
Head_Tail_Sets_Results('../generated_predictions/revision_rec/CodeBERT/CE/', vocab_tokens, freq_vocab, threshold, 'codebert')
Head_Tail_Sets_Results('../generated_predictions/revision_rec//T5_review/CE/', vocab_tokens, freq_vocab, threshold, 't5_review')

