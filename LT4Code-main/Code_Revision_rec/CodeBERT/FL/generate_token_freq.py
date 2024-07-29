

from transformers import RobertaTokenizer
from collections import Counter
import math
from tqdm import tqdm
import pickle
import json

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

splits = ['train', 'val', 'test']

all_targets = []
for split in splits:
    filename = '../data/'+split+'.json'

    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=' '.join(js['code_tokens']).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())
            target = nl
            all_targets.append(target)



vocab = Counter()
for l in tqdm(all_targets):

    target_tokens = tokenizer.tokenize(l)[:400]
    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    vocab.update(target_ids)






vocab_tokens = [i[0] for i in vocab.most_common(len(vocab))]
vocab_samples = [i[1] for i in vocab.most_common(len(vocab))]
top_total = sum(i[1] for i in vocab.most_common(len(vocab)))
vocab_score = [ 1/math.sqrt(n) for n in vocab_samples]


all_token_freq = []
for i in range(50265):
    if i in vocab.keys():
        all_token_freq.append(vocab[i])
    elif i == 1:
        all_token_freq.append(int(top_total/50265))
    else:
        all_token_freq.append(1)
print()

with open('../data/codebert_token_freq_file.pkl', 'wb') as f:
    pickle.dump(all_token_freq, f)

