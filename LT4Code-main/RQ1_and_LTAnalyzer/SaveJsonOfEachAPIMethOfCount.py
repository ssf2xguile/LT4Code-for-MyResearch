import json
from collections import Counter
import pandas as pd


def calculate_label_freq(task, data_dir):
    if task == 'api':
        # API class Data
        splits = ['train']
        all_data = []
        
        for split in splits:
            file_name = data_dir + split + '_3_lines.csv'
            df = pd.read_csv(file_name)
            all_data.append(df)
        
        all_data = pd.concat(all_data)
        all_data = list(all_data['target_api'])

        all_api_data = []

        def parse_string_into_apis(str_):
            apis = []
            eles = str_.split('.')
            first_lib = eles[0]

            for i in range(1, len(eles) - 1):
                try:
                    module_, library_ = eles[i].strip().rsplit(' ')
                except:
                    module_, library_ = eles[i].strip().split(' ', 1)
                apis.append((first_lib.strip() + '.' + module_.strip()).replace(' ', ''))
                first_lib = library_

            apis.append((first_lib.strip() + '.' + eles[-1].strip()).replace(' ', ''))
            return apis

        for d in all_data:
            api_seqs = parse_string_into_apis(d)
            api_seqs = [a.lower() for a in api_seqs]
            all_api_data.extend(api_seqs)

        vocab = Counter(all_api_data)
        vocab_tokens = [i[0] for i in vocab.most_common(len(vocab))]
        vocab_samples = [i[1] for i in vocab.most_common(len(vocab))]

        api_rec_df = {
            'id': list(range(len(vocab_tokens))),
            'API Method': vocab_tokens,
            'count': vocab_samples
        }

        df = pd.DataFrame(api_rec_df)

        # Sort by count in descending order
        df = df.sort_values(by='count', ascending=False).reset_index(drop=True)

        # Update id after sorting
        df['id'] = df.index

        # Determine the id up to which the top 50% API methods fall
        total_count = df['count'].sum()
        cumulative_count = df['count'].cumsum()
        top_50_percent_index = (cumulative_count <= total_count * 0.5).sum() - 1

        # Print the id representing the top 50% API methods
        print(f'The id up to which the top 50% of API methods are covered: {top_50_percent_index}')

        # Jsonファイルを保存する。初回のみ実行
        #output_file = './api_rec_data.json'
        #df.to_json(output_file, orient='records', force_ascii=False, indent=4)


# --task=api --data_dir='../all_data/api_seq_data/mularec_data/'
task = 'api'
data_dir = '../all_data/api_seq_data/mularec_data/'
calculate_label_freq(task, data_dir)
