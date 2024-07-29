
import pandas as pd
import numpy as np

split = 'test' ##'validate'  'train' 'test'
filename = '../data/'+split+'_3_lines.csv'
df = pd.read_csv(filename)
df = df.fillna("")
code = df['source_code'].astype("string").tolist()
ant = df['annotation'].astype("string").tolist()
post = df['related_so_question'].astype("string").tolist()
# post = df['similar_post'].astype("string").tolist()
api_seq = df['target_api'].astype("string").tolist()


output_dir = '../data/codet5_format_data/refine/small/'
output_file_src = output_dir + split+'.buggy-fixed.buggy'
output_file_tgt = output_dir + split+'.buggy-fixed.fixed'


targets = [l.lower() for l in api_seq]
sources = []
for i in range(len(code)):
    input_ =  'text: '+ ant[i].strip().lower().replace('\n', ' ') + ' \t code: '+ code[i].lower().replace('\n', ' ')
    sources.append(input_)

print()

with open(output_file_src, 'w') as f:
    for l in sources:
        f.write(l.strip()+'\n')
with open(output_file_tgt, 'w') as f:
    for l in targets:
        f.write(l.strip()+'\n')


"""
查看targets的平均长度多长
"""
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-base")

print(len(targets))
lengths = []
for l in targets:
    #engths.append(len(l.split()))
    lengths.append(len(tokenizer(l)['input_ids']))

hist, bins = np.histogram(lengths, bins=5, range=(0,500))
print(hist)
print(bins)


""" 93%的数据labels都在100个subtoken以下所以target seq就取100了  """