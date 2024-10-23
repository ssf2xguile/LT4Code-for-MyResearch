# 17_test_exactmatch_whole_byTargetInvestigate.pyとほとんど似ている。違うのはparse_string_into_apisの返り値が単一の文字列であるという点。
# APIメソッドシーケンスの最初のAPIメソッドが一致しているかどうかを比較する。比較して一致していた場合、そのAPIメソッドがテールかどうかを判定する。
# 出力: 正解ヘッドAPIメソッドのインデックスを記述したテキストファイルと正解テールAPIメソッドのインデックスを記述したテキストファイル。
import pandas as pd
import argparse
import json
import os

def compare_apimethod(model_preds, correct_refs, api_dict):
    correct_head_index = []
    correct_tail_index = []
    for index, (pred, ref) in enumerate(zip(model_preds, correct_refs)):
        if pred == ref:
            if api_dict[ref]['head_or_tail'] == 0:
                correct_head_index.append(index)
            else:
                correct_tail_index.append(index)
    return correct_head_index, correct_tail_index

def parse_string_into_apis(str_):   # 今回は小文字化と空白削除の処理を追加して、さらに最初の1個のAPIメソッドだけしか返さない。つまり、文字列が返ってくる。
    apis = []
    eles = str_.split('\t')[0].strip().split('.')

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
    return apis[0]  

def write_matching_indices(file_path, indices):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index in indices:
            file.write(f"{index}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='../data/CodeBERT_predictions.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='../data/CodeT5_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='../data/MulaRec_predictions.txt', help="Path to save the output file")
    parser.add_argument('--investigate_file', type=str, default='../data/target_investigate.csv', help="Path to save the output file")
    parser.add_argument('--api_database', type=str, default='../data/first_correctCount_and_appearCount.json', help="whole")
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='../data/', help="Path to save the text file which including tail data")
    args = parser.parse_args()

    # 予測結果を読み込む
    with open(args.file1, 'r', encoding='utf-8') as file1:
        codebert_preds_file = [parse_string_into_apis(pred) for pred in file1.readlines()]
    with open(args.file2, 'r', encoding='utf-8') as file2:
        codet5_preds_file = [parse_string_into_apis(pred) for pred in file2.readlines()]    # [...,'base64.decodetoobject, 'math.sqrt', 'thread.start',...]のようになる 
    with open(args.file3, 'r', encoding='utf-8') as file3:
        mularec_preds_file = [parse_string_into_apis(pred) for pred in file3.readlines()]   # [...,'base64.decodetoobject', 'point.math', 'thread.start',...]のようになる

    # 調査対象データに該当する予測結果のインデックスのリストを取り出す
    df = pd.read_csv(args.investigate_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()  # この処理によってリストのリストが得られる[...,'base64.decodetoobject', 'math.sqrt', 'thread.start',...]
    original_index = df['api_index'].tolist()   # 元のテストデータの何番目のデータかを示すインデックス[..., 36149, 36154, ...]のようになる 

    # APIデータベースを読み込む
    with open(args.api_database, 'r', encoding='utf-8') as file:
            api_database = json.load(file)
    # APIメソッド名をキー、詳細を値とする辞書に変換
    api_dict = {entry["api_method"]: entry for entry in api_database}
    
    codebert_preds = []
    codet5_preds = []
    mularec_preds = []
    for index in original_index:
        codebert_preds.append(codebert_preds_file[int(index)-1])
        codet5_preds.append(codet5_preds_file[int(index)-1])
        mularec_preds.append(mularec_preds_file[int(index)-1])

    # 各モデルの正解数をカウントする
    codebert_correct_head_index, codebert_correct_tail_index = compare_apimethod(codebert_preds, correct_refs, api_dict)
    codet5_correct_head_index, codet5_correct_tail_index = compare_apimethod(codet5_preds, correct_refs, api_dict)
    mularec_correct_head_index, mularec_correct_tail_index = compare_apimethod(mularec_preds, correct_refs, api_dict)

    # 最終結果の出力
    codebert_output_file1 = os.path.join(args.output_dir, f'CodeBERT_first_exactmatch_index_head.txt')
    codebert_output_file2 = os.path.join(args.output_dir, f'CodeBERT_first_exactmatch_index_tail.txt')
    write_matching_indices(codebert_output_file1, codebert_correct_head_index)
    write_matching_indices(codebert_output_file2, codebert_correct_tail_index)
    print(f"CodeBERT correct count data has been saved to {codebert_output_file1}")
    print(f"CodeBERT correct head api count: {len(codebert_correct_head_index)}")
    print(f"CodeBERT correct tail api count: {len(codebert_correct_tail_index)}")
    codet5_output_file1 = os.path.join(args.output_dir, f'CodeT5_first_exactmatch_index_head.txt')
    codet5_output_file2 = os.path.join(args.output_dir, f'CodeT5_first_exactmatch_index_tail.txt')
    write_matching_indices(codet5_output_file1, codet5_correct_head_index)
    write_matching_indices(codet5_output_file2, codet5_correct_tail_index)
    print(f"CodeT5 correct count data has been saved to {codet5_output_file1}")
    print(f"CodeT5 correct head api count: {len(codet5_correct_head_index)}")
    print(f"CodeT5 correct tail api count: {len(codet5_correct_tail_index)}")
    mularec_output_file1 = os.path.join(args.output_dir, f'MulaRec_first_exactmatch_index_head.txt')
    mularec_output_file2 = os.path.join(args.output_dir, f'MulaRec_first_exactmatch_index_tail.txt')
    write_matching_indices(mularec_output_file1, mularec_correct_head_index)
    write_matching_indices(mularec_output_file2, mularec_correct_tail_index)
    print(f"MulaRec correct count data has been saved to {mularec_output_file1}")
    print(f"MulaRec correct head api count: {len(mularec_correct_head_index)}")
    print(f"MulaRec correct tail api count: {len(mularec_correct_tail_index)}")


if __name__ == "__main__":
    main()