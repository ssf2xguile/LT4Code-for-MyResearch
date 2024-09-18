import json
from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import json
import argparse
import os
from time import time
import json
from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import json

def gini_coef(wealths):
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float64(len(cum_wealths)-1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return (A / (A+B))


def calculate_label_freq(task, data_dir):

    if task == 'vulnerability':

        ###TreeVul Data
        splits = ['train', 'test','validation']
        splits = ['train']
        seen_msgs= []
        all_data = []
        for split in splits:
            file_name = data_dir + split +'_set.json'
            with open (file_name, 'r') as f:
                d_multi_sample = json.load(f)
                for d in d_multi_sample:
                    if d['commit_id'] not in seen_msgs:
                        all_data.append(d)
                        seen_msgs.append(d['commit_id'])

        treevul_ids = list(range(len(all_data)))
        treevul_labels = [d['cwe_list']  for d in all_data]
        treevul_categories = ['vulnerability_type']*len(all_data)
        vocab = Counter(treevul_labels)
        vocab_tokens = [i[0] for i in vocab.most_common(len(vocab))]
        vocab_samples = [i[1] for i in vocab.most_common(len(vocab))]
        total_smaple = sum(vocab_samples)


        label2label_num_dict = {}
        for i in range(len(vocab_tokens)):
            label2label_num_dict[vocab_tokens[i]] = i
        treevul_labels = [ label2label_num_dict[l] for l in treevul_labels]
        treevul_df = {'id': treevul_ids, 'Sorted ID': treevul_labels, 'task': treevul_categories   }
        df = pd.DataFrame(treevul_df)

        vocab_taking_up_ratios = [float(l / total_smaple) for l in vocab_samples]
        print('\n****** Gini coeffcient:', gini_coef(vocab_taking_up_ratios))


        
        # # Draw Plot
        fig_scale = 0.4
        plt.figure(figsize=(int(20*fig_scale),int(10*fig_scale)), dpi= int(80/fig_scale))
        sns.distplot(df.loc[df['task'] == 'vulnerability_type', "Sorted ID"], color="dodgerblue", label="vulnerability_type",  hist_kws={'alpha':.7}, kde_kws={'linewidth':2})
        plt.xlim(0, )
        plt.xlabel('Sorted Label ID', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.tight_layout()
        plt.show()
        


    elif task == 'revision':
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
            longpath = s.find_longest_match(0, len(det),  0, len(add))
            matching = s.get_matching_blocks()

            for tag, i1, i2, j1, j2 in s.get_opcodes():
                if tag == 'delete':
                    for src_token in det[i1:i2]:
                        all_edits.append( (src_token, '')  )
                elif tag == 'insert':
                    for tgt_token in add[j1:j2]:
                        all_edits.append( ('', tgt_token)  )
                elif tag == 'replace':
                    for src_token in det[i1:i2]:
                        all_edits.append( (src_token, '')  )
                    for tgt_token in add[j1:j2]:
                        all_edits.append( ('', tgt_token)  )
            return all_edits


        splits = ['train', 'test','valid']
        splits = ['train']
        all_sources, all_golds = [],[]
        for split in splits:
            file_name = data_dir.rsplit('/', 2)[0] + '/raw/' + 'src-' + split + '.txt'
            with open (file_name, 'r') as f:
                sources = f.readlines()
                all_sources.extend(sources)
            file_name = data_dir + split + '.buggy-fixed.fixed'
            with open(file_name, 'r') as f:
                golds = f.readlines()
                all_golds.extend(golds)


        all_edits_all_data = []
        for i in tqdm(range(len(all_sources))):
            all_edits_all_data.extend(get_token_level_edit_pair(all_sources[i], all_golds[i]))
        all_data = all_edits_all_data

      
        revision_ids = list(range(len(all_data)))
        revision_labels = all_data
        revision_categories = ['code_revision']*len(all_data)
        vocab = Counter(revision_labels)
        vocab_tokens = [i[0] for i in vocab.most_common(len(vocab))]
        vocab_samples = [i[1] for i in vocab.most_common(len(vocab))]
        total_smaple = sum(vocab_samples)
        label2label_num_dict = {}
        for i in range(len(vocab_tokens)):
            label2label_num_dict[vocab_tokens[i]] = i
        revision_labels = [ label2label_num_dict[l] for l in revision_labels]
        revision_df = {'id': revision_ids, 'Sorted ID': revision_labels, 'task': revision_categories   }
        df = pd.DataFrame(revision_df)


        
        # # Draw Plot
        fig_scale = 0.4
        plt.figure(figsize=(int(20*fig_scale),int(10*fig_scale)), dpi= int(80/fig_scale))
        sns.distplot(df.loc[df['task'] == 'code_revision', "Sorted ID"], color="orange", hist_kws={'alpha':.7}, kde_kws={'linewidth':2})
        plt.ylim(0, 0.0012)
        plt.xlim(0, )
        plt.xlabel('Sorted Label ID', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.tight_layout()
        plt.show()
        

        vocab_taking_up_ratios = [ round(float(l / total_smaple),5) for l in vocab_samples]
        print('Gini coeffcient:', gini_coef(vocab_taking_up_ratios))




    elif task == 'api':

        ###API class Data
        splits = ['train', 'test','validate']
        splits = ['train']
        all_data = []
        for split in splits:
            file_name = data_dir + split + '_3_lines.csv'
            df = pd.read_csv(file_name)
            all_data.append(df)
            print(len(df))
        all_data = pd.concat(all_data)
        all_data = list(all_data['target_api'])
        print(len(all_data))

        all_api_data = []
        def parse_string_into_apis(str_):
            apis = []
            eles = str_.split('.')

            first_lib = eles[0]

            for i in range(1, len(eles)-1):
                try:
                    module_, library_ = eles[i].strip().rsplit(' ')
                except:
                    module_, library_ = eles[i].strip().split(' ', 1)
                apis.append(first_lib.strip()+'.'+module_.strip())
                first_lib = library_

            apis.append(first_lib.strip() +'.'+ eles[-1].strip())
            return apis

        for d in all_data:
            api_seqs = parse_string_into_apis(d)
            api_seqs = [a.lower() for a in api_seqs]
            all_api_data.extend(api_seqs)
        all_data = all_api_data


        api_rec_ids = list(range(len(all_data)))
        api_rec_labels = all_data
        api_rec_categories = ['api_sequence_rec']*len(all_data)
        vocab = Counter(api_rec_labels)
        vocab_tokens = [i[0] for i in vocab.most_common(len(vocab))]
        vocab_samples = [i[1] for i in vocab.most_common(len(vocab))]
        total_smaple = sum(vocab_samples)

        # APIメソッドの種類数を表示
        print(f"APIメソッドの種類数: {len(vocab_tokens)}")
        label2label_num_dict = {}
        for i in range(len(vocab_tokens)):
            label2label_num_dict[vocab_tokens[i]] = i
        api_rec_labels = [ label2label_num_dict[l] for l in api_rec_labels]
        api_rec_df = {'id': api_rec_ids, 'Sorted ID': api_rec_labels, 'task': api_rec_categories   }
        df = pd.DataFrame(api_rec_df)

        
        # # Draw Plot
        fig_scale = 0.4
        plt.figure(figsize=(int(20*fig_scale),int(10*fig_scale)), dpi= int(80/fig_scale))
        sns.distplot(df.loc[df['task'] == 'api_sequence_rec', "Sorted ID"], color=sns.color_palette("husl", 8)[0], label="api_sequence_rec",  hist_kws={'alpha':.7}, kde_kws={'linewidth':2})
        plt.ylim(0, 0.00045)
        plt.xlim(0, )
        plt.xlabel('Sorted Label ID', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.tight_layout()
        plt.show()
        

        vocab_taking_up_ratios = [float(l / total_smaple) for l in vocab_samples]
        print('Gini coeffcient:', gini_coef(vocab_taking_up_ratios))


# --task=api --data_dir='../all_data/api_seq_data/mularec_data/'
# --task=revision --data_dir='../all_data/code_review_data/codet5_data/codet5_format_data/refine/small/'
# --task=vulnerability --data_dir='../all_data/vulnerability_data/dataset/'
task='api'
data_dir='../all_data/api_seq_data/mularec_data/'
calculate_label_freq(task, data_dir)
