# 3つのLLMによるExactmatchしたテストデータのうち、テールデータを含むテストデータのインデックスを洗い出す
import pandas as pd
import argparse
import json
import os

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 2個以上の連続する空白を1個の空白に置き換える
    lines = [re.sub(r'\s+', ' ', line) for line in lines]
    return lines

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    lines = [' '.join(item['docstring_tokens']) for item in data]
    # 2個以上の連続する空白を1個の空白に置き換える
    lines = [re.sub(r'\s+', ' ', line) for line in lines]
    return lines

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df['target_api'].str.lower()  # Exclude missing values

def check_if_tail(api, api_database):
    """JSONデータベースからAPIが存在するかどうかを確認し、テールかヘッドかをチェック"""
    for entry in api_database:
        if entry['api_method'] == api:
            return entry['head_or_tail'] == 1  # True if it's a tail (head_or_tail=1)
    return None  # APIがデータベースに存在しない場合

def parse_string_into_apis(str_):   # 今回は小文字化と空白削除の処理を追加している。
    apis = []
    eles = str_.split('.')

    first_lib = eles[0]

    for i in range(1, len(eles)-1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
            api = first_lib.strip()+'.'+module_.strip()
            api = api.lower().replace(' ','')
            apis.append(api)
            first_lib = library_
        except ValueError:
            try:
                module_, library_ = eles[i].strip().split(' ', 1)
                api = first_lib.strip()+'.'+module_.strip()
                api = api.lower().replace(' ','')
                apis.append(api)
                first_lib = library_
            except ValueError:  # splitで失敗した場合の処理 例えばPoint . Math . pow Math . sqrtのようにドットが繋がっている場合
                module_ = eles[i].strip()
                library_ = ''
                api = first_lib.strip()+'.'+module_.strip()
                api = api.lower().replace(' ','')
                apis.append(api)
                first_lib = module_

    api = first_lib.strip()+'.'+eles[-1].strip()
    api = api.lower().replace(' ','')
    apis.append(api)
    return apis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='../data/CodeBERT_exactmatch_index.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='../data/CodeT5_exactmatch_index.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='../data/MulaRec_exactmatch_index.txt', help="Path to save the output file")
    parser.add_argument('--test_file', type=str, default='../data/test_3_lines.csv', help="Path to save the output file")
    parser.add_argument('--head_or_tail_database', type=str, default='../data/api_head_or_tail.json', help="Path to save the output file")
    parser.add_argument('--output_dir', type=str, default='../data/', help="Path to save the text file which including tail data")
    args = parser.parse_args()

    with open(args.file1, 'r', encoding='utf-8') as file1:
        index1 = file1.readlines()
    with open(args.file2, 'r', encoding='utf-8') as file2:
        index2 = file2.readlines()
    with open(args.file3, 'r', encoding='utf-8') as file3:
        index3 = file3.readlines()

    # JSONデータベースを読み込む
    with open(args.head_or_tail_database, 'r', encoding='utf-8') as file:
        api_database = json.load(file)

    # 正解データを加工する
    df = pd.read_csv(args.test_file)
    api_list = df['target_api'].apply(parse_string_into_apis).tolist()  # この処理によってリストのリストが得られる[...,['error.<init>'], ['util.contains', 'message.src'], ['string.length'],...]

    # 結果を格納する配列
    tail_api_indices1 = []
    tail_api_indices2 = []
    tail_api_indices3 = []

    for idx1 in index1:                                                 # CodeBERT
        api_seq = api_list[int(idx1)-1]                                 # テキストから読み込んだインデックスをもとに、テストデータを検索する
        for api in api_seq:
            is_tail = check_if_tail(api, api_database)
            if is_tail is not None and is_tail:  # APIがテールなら処理を終了してインデックスを追加
                tail_api_indices1.append(int(idx1))
                break  # 次のインデックスに進む
    print(len(tail_api_indices1))

    for idx2 in index2:                                                 # CodeT5
        api_seq = api_list[int(idx2)-1]                                 # テキストから読み込んだインデックスをもとに、テストデータを検索する
        for api in api_seq:
            is_tail = check_if_tail(api, api_database)
            if is_tail is not None and is_tail:  # APIがテールなら処理を終了してインデックスを追加
                tail_api_indices2.append(int(idx2))
                break  # 次のインデックスに進む
    print(len(tail_api_indices2))

    for idx3 in index3:                                                 # MulaRec
        api_seq = api_list[int(idx3)-1]                                 # テキストから読み込んだインデックスをもとに、テストデータを検索する
        for api in api_seq:
            is_tail = check_if_tail(api, api_database)
            if is_tail is not None and is_tail:  # APIがテールなら処理を終了してインデックスを追加
                tail_api_indices3.append(int(idx3))
                break  # 次のインデックスに進む
    print(len(tail_api_indices3))

    # 結果をテキストファイルに保存する
    with open(os.path.join(args.output_dir,'CodeBERT_exactmatch_index_includingtail.txt'), 'w', encoding='utf-8') as f1:
        for idx in tail_api_indices1:
            f1.write(f"{idx}\n")
    # 結果をテキストファイルに保存する
    with open(os.path.join(args.output_dir,'CodeT5_exactmatch_index_includingtail.txt'), 'w', encoding='utf-8') as f2:
        for idx in tail_api_indices2:
            f2.write(f"{idx}\n")
    # 結果をテキストファイルに保存する
    with open(os.path.join(args.output_dir,'MulaRec_exactmatch_index_includingtail.txt'), 'w', encoding='utf-8') as f3:
        for idx in tail_api_indices3:
            f3.write(f"{idx}\n")

if __name__ == "__main__":
    main()