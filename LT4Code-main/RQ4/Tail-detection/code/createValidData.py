"""
テール判定用の検証データを作成するためのプログラム。
ヘッドAPIメソッドが分布の上位何%であるかを定義した情報に基づいて、APIメソッドシーケンスを精査し100%ヘッドAPIメソッドを占めていれば、そのAPIメソッドシーケンスに紐づくもとのデータをヘッドとする。それ以外をテールとし、新しいデータセットを作成する。
目的に合わせて変更すべき変更の値は3つ
74行目: 212 これはAPIメソッドの出現頻度分布のうち何個目のAPIメソッドまでがヘッドAPIメソッドであるのかを表している。HeadAndTailAnalyze.pyを実行することでこの値は得られる。
81行目: 100  これはAPIメソッドシーケンス中何%以上ヘッドAPIメソッドを占めていれば、元のデータをヘッドとして判定すべきかを表す値である。
105行目: api_data_30_headAPIMethod
"""
import argparse
import json
import os

# Function to parse API methods from file1
def parse_string_into_apis(str_):
    apis = []
    eles = str_.split('\t')[0].strip().split('.')
    first_lib = eles[0]

    for i in range(1, len(eles) - 1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
        except ValueError:
            try:
                module_, library_ = eles[i].strip().split(' ', 1)
            except ValueError:
                module_ = eles[i].strip()
                library_ = ''

        api = first_lib.strip() + '.' + module_.strip()
        api = api.lower().replace(' ', '')
        apis.append(api)
        first_lib = library_ if library_ else module_

    api = first_lib.strip() + '.' + eles[-1].strip()
    api = api.lower().replace(' ', '')
    apis.append(api)
    return apis

# Process file1
def process_file1(filepath):
    with open(filepath, 'r') as f:
        return [parse_string_into_apis(line) for line in f]

# Process file2
def process_file2(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

# Process file3
def process_file3(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {entry['API Method']: entry for entry in data}

# Main analysis function
def analyze_data(file1_data, file2_data, file3_dict, output_dir):
    results = []
    new_valid_data = []

    for entry in file2_data:
        idx = entry['idx'] - 295995 - 37000    # 訓練データ数とテストデータ数を引かないといけない
        if idx >= len(file1_data):
            continue

        api_sequence = file1_data[idx]
        head_count, total_count = 0, 0

        for api in api_sequence:
            api_info = file3_dict.get(api)
            if api_info:
                total_count += 1
                if api_info['id'] <= 11439:    # 46で上位30%, 212で上位50%, 964で上位70%, 11439で上位90%
                    head_count += 1

        if total_count > 0:
            head_ratio = round((head_count / total_count) * 100, 3)
            results.append((api_sequence, head_ratio))
            new_valid_data.append({
                "target": 0 if head_ratio >= 100 else 1,
                "func": entry["func"],
            })
        else:
            head_ratio = 0
            results.append((api_sequence, head_ratio))
            new_valid_data.append({
                "target": 0 if head_ratio >= 100 else 1,
                "func": entry["func"],
            })    
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save new validing data
    output_file = os.path.join(output_dir, "valid.jsonl")
    with open(output_file, "w") as f:
        for entry in new_valid_data:
            f.write(json.dumps(entry) + "\n")

# Main function
def main():
    parser = argparse.ArgumentParser(description='Process API method data.')
    parser.add_argument('--file1', type=str, default='../../../all_data/api_seq_data/codet5_data/codet5_format_data/refine/small/valid.buggy-fixed.fixed')
    parser.add_argument('--file2', type=str, default='../../../all_data/RQ4_data/api_data/valid.jsonl')
    parser.add_argument('--file3', type=str, default='../../../RQ1_and_LTAnalyzer/api_rec_data.json')
    parser.add_argument('--output_dir', type=str, default='../../../all_data/RQ4_data/api_data_90_headAPIMethod')
    args = parser.parse_args()

    file1_data = process_file1(args.file1)
    file2_data = process_file2(args.file2)
    file3_dict = process_file3(args.file3)

    analyze_data(file1_data, file2_data, file3_dict, args.output_dir)

if __name__ == '__main__':
    main()
