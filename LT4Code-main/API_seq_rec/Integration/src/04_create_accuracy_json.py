import json
from collections import Counter
import pandas as pd
import argparse
import os

def parse_string_into_apis(str_):
    apis = []
    eles = str_.split('.')

    first_lib = eles[0]

    for i in range(1, len(eles)-1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
        except:
            module_, library_ = eles[i].strip().split(' ', 1)
        apis.append(first_lib.strip() + '.' + module_.strip())
        first_lib = library_

    apis.append(first_lib.strip() + '.' + eles[-1].strip())
    return apis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='../data/CodeBERT_exactmatch_index_includingtail.txt', help="Path to the CodeBERT exactmatch file")
    parser.add_argument('--file2', type=str, default='../data/CodeT5_exactmatch_index_includingtail.txt', help="Path to the CodeT5 exactmatch file")
    parser.add_argument('--file3', type=str, default='../data/MulaRec_exactmatch_index_includingtail.txt', help="Path to the MulaRec exactmatch file")
    parser.add_argument('--test_file', type=str, default='../data/test_3_lines.csv', help="Path to the test data file")
    parser.add_argument('--output_dir', type=str, default='../data/', help="Path to save the output JSON file")
    args = parser.parse_args()

    # テストファイルを読み込んでAPIのリストを取得
    df = pd.read_csv(args.test_file)
    all_api_data = []
    all_api_data_with_structure = []
    for api_sequence in df['target_api']:
        api_seqs = parse_string_into_apis(api_sequence)
        api_seqs = [a.lower().replace(' ', '') for a in api_seqs]
        all_api_data.extend(api_seqs)           # 最終的にこうなる['hashset.<init>', 'collection.isempty', 'collection<integer>.add',...,'map<string,string>.put']
        all_api_data_with_structure.append(api_seqs)  #最終的にこうなる[...,['error.<init>'], ['util.contains', 'message.src'], ['string.length'],...]

    # APIの出現頻度をカウント
    vocab = Counter(all_api_data)
    # 各モデルの正解APIインデックスを読み込む
    with open(args.file1, 'r') as file1, open(args.file2, 'r') as file2, open(args.file3, 'r') as file3:
        codebert_indices = [int(idx.strip()) for idx in file1.readlines()]
        codet5_indices = [int(idx.strip()) for idx in file2.readlines()]
        mularec_indices = [int(idx.strip()) for idx in file3.readlines()]

    # 各APIについて正解回数を初期化
    api_results = {}
    for api_method, appearance_count in vocab.items():
        api_results[api_method] = {
            "api_method": api_method,
            "appearance_count": appearance_count,
            "CodeBERT_correct_count": 0,
            "CodeT5_correct_count": 0,
            "MulaRec_correct_count": 0
        }

    # CodeBERT、CodeT5、MulaRecの正解数をカウント
    for idx in codebert_indices:
        api_sequence = all_api_data_with_structure[idx - 1]  # インデックスは1始まりなので-1する
        for api_method in api_sequence:
            if api_method in api_results:
                api_results[api_method]["CodeBERT_correct_count"] += 1

    for idx in codet5_indices:
        api_sequence = all_api_data_with_structure[idx - 1]  # インデックスは1始まりなので-1する
        for api_method in api_sequence:
            if api_method in api_results:
                api_results[api_method]["CodeT5_correct_count"] += 1

    for idx in mularec_indices:
        api_sequence = all_api_data_with_structure[idx - 1]  # インデックスは1始まりなので-1する
        for api_method in api_sequence:
            if api_method in api_results:
                api_results[api_method]["MulaRec_correct_count"] += 1

    # 出力ファイルに書き込むためにリストに変換し、登場回数が多い順にソート
    output_list = sorted(api_results.values(), key=lambda x: x['appearance_count'], reverse=True)

    # 結果をJSONファイルに保存
    output_file = os.path.join(args.output_dir, 'api_correct_count.json')
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(output_list, json_file, indent=4)

    print(f"API correct count data has been saved to {output_file}")

if __name__ == "__main__":
    main()
