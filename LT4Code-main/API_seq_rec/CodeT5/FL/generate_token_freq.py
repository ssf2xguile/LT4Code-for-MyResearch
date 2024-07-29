

from transformers import RobertaTokenizer
from collections import Counter
import math
from tqdm import tqdm
import pickle

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

splits = ['train', 'valid', 'test']

all_targets = []
for split in splits:
    file_path = '../data/codet5_format_data/refine/small/'+split+'.buggy-fixed.fixed'
    with open(file_path, 'r') as f:
        all_targets.extend( f.readlines() )

vocab = Counter()
for l in tqdm(all_targets):
    #vocab.update(tokenizer(l)['input_ids'])
    target_str = l.replace('</s>', '<unk>')
    target_ids = tokenizer.encode(target_str, max_length=400, truncation=True)
    vocab.update(target_ids)

vocab_tokens = [i[0] for i in vocab.most_common(len(vocab))]
vocab_samples = [i[1] for i in vocab.most_common(len(vocab))]
top_total = sum(i[1] for i in vocab.most_common(len(vocab)))
vocab_score = [ 1/math.sqrt(n) for n in vocab_samples]


all_token_freq = []
for i in range(32100):
    if i in vocab.keys():
        all_token_freq.append(vocab[i])
    elif i == 0:
        all_token_freq.append(int(top_total/32100))
    else:
        all_token_freq.append(1)
print()

with open('../data/codet5_format_data/refine/small/token_freq_file.pkl', 'wb') as f:
    pickle.dump(all_token_freq, f)

