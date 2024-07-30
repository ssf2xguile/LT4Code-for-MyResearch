"""
生成したAPIメソッドシーケンスの正解率を調査する。
正解率=前方一致するAPIメソッドの個数/正解APIメソッドシーケンス中に含まれるAPIメソッドの個数
入力: 各モデルによるAPIメソッドシーケンス生成結果のテキストファイル, 正解APIメソッドシーケンスのテキストファイル
出力: 全体の正解率(コマンドライン表示), 個々の正解率(図が作成される)
"""
import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return [' '.join(item['docstring_tokens']) for item in data]

def read_fixed_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df['target_api'].str.lower()  # Convert to lowercase, because ground trouth api sequence is presented in Capital character.

def check_matching_lines(txt_lines, test_lines):    #txt_lines:予測結果のAPIメソッドのリスト["Utility . getValidChoice ... Scanner . nextLine", "Error . <init> ... ExceptionInInitializer . error", ...], test_lines:テストデータのAPIメソッドのリスト["HashSet . <init> ... String . charAt", ]
    accuracy_list = []
    all_txt_apimethod_list = [] #予測結果のAPIメソッドのリストを格納するリスト [["utility . getvalidchoice", ... "scanner . nextline"], ["error . <init>", ... "exceptionininitializer . error"], ...]
    all_test_apimethod_list = [] #テストデータのAPIメソッドのリストを格納するリスト [["hashset . <init>", ... "string . charat"], ...]
    for i, (txt_line, test_line) in enumerate(zip(txt_lines, test_lines)):
        txt_apimethod_list = parse_string_into_apis(txt_line)
        test_apimethod_list = parse_string_into_apis(test_line)
        txt_apimethod_list = [a.lower() for a in txt_apimethod_list]
        test_apimethod_list = [a.lower() for a in test_apimethod_list]
        all_txt_apimethod_list.append(txt_apimethod_list)
        all_test_apimethod_list.append(test_apimethod_list)
    for i, (txt_apimethod, test_apimethod) in enumerate(zip(all_txt_apimethod_list, all_test_apimethod_list)):
        matching_count = 0
        for j in range(len(test_apimethod)):
            try:
                if txt_apimethod[j] == test_apimethod[j]:
                    matching_count += 1
                else:
                    break
            except IndexError:
                break
        indivisuial_accuracy = round(matching_count / len(test_apimethod), 2)
        accuracy_list.append(indivisuial_accuracy)

    return accuracy_list

def parse_string_into_apis(str_):
    apis = []
    eles = str_.split('.')

    first_lib = eles[0]

    for i in range(1, len(eles)-1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
            apis.append(first_lib.strip()+'.'+module_.strip())
            first_lib = library_
        except ValueError:
            try:
                module_, library_ = eles[i].strip().split(' ', 1)
                apis.append(first_lib.strip()+'.'+module_.strip())
                first_lib = library_
            except ValueError:  # splitで失敗した場合の処理 例えばPoint . Math . pow Math . sqrtのようにドットが繋がっている場合
                module_ = eles[i].strip()
                library_ = ''
                apis.append(first_lib.strip()+'.'+module_.strip())
                first_lib = module_

    apis.append(first_lib.strip() +'.'+ eles[-1].strip())
    return apis

def write_matching_indices(file_path, indices):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index in indices:
            file.write(f"{index}\n")

def plot_accuracy_distribution(accuracy_list, output_dir):
    # 横軸のラベルを定義
    bins = [i/10 for i in range(11)]
    labels = ['0%-10%', '10%-20%', '20%-30%', '30%-40%', '40%-50%', '50%-60%', '60%-70%', '70%-80%', '80%-90%', '90%-100%']

    # 各バケットの個数をカウント
    counts, _ = np.histogram(accuracy_list, bins=bins)

    # プロットを作成
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, width=0.8, align='center')
    plt.xlabel('Accuracy Range')
    plt.ylabel('Number of API Methods')
    plt.title('Distribution of API Method Accuracy')
    plt.savefig(os.path.join(output_dir, "accuracy_distribution.png"), format='png', dpi=200)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str, default='../data/MulaRec_predictions.txt',help="Path to the prediction.txt file")
    parser.add_argument('--test_file', type=str, default='../data/test_3_lines.csv', help="Path to the test.json file")
    parser.add_argument('--output_dir', type=str, default='../data/', help="Path to save the output file")
    args = parser.parse_args()

    txt_lines = read_txt_file(args.prediction_file)

    # Determine the file extension and read accordingly
    _, ext = os.path.splitext(args.test_file)
    if ext == '.json':
        test_lines = read_json_file(args.test_file)
    elif ext == '.fixed':
        test_lines = read_fixed_file(args.test_file)
    elif ext == '.csv':
        test_lines = read_csv_file(args.test_file)
    else:
        raise ValueError("Unsupported file extension. Only .json and .fixed files are supported.")

    accuracy_list = check_matching_lines(txt_lines, test_lines)

    total_accuracy = sum(accuracy_list) / len(accuracy_list)
    print(f"total_accuracy: {total_accuracy}")

    plot_accuracy_distribution(accuracy_list, args.output_dir)

if __name__ == "__main__":
    main()