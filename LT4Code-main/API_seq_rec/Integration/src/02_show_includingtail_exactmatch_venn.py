# 3つのLLMによるExactmatchしたテストデータのうち、テールデータを含むテストデータについてのベン図を示す
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import pandas as pd
import argparse

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

def read_fixed_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df['target_api'].str.lower()  # Exclude missing values

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='../data/CodeBERT_exactmatch_index.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='../data/CodeT5_exactmatch_index.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='../data/MulaRec_exactmatch_index.txt', help="Path to save the output file")
    parser.add_argument('--test_file', type=str, default='../data/test_3_lines.csv', help="Path to save the output file")
    parser.add_argument('--head_or_tail_database', type=str, default='../data/api_head_or_tail.json', help="Path to save the output file")
    parser.add_argument('--output', type=str, default='../data/Exactmatch_includingtail_venn_diagram.png', help="Path to save the Venn diagram image")
    parser.add_argument('--dpi', type=int, default=300, help="DPI for the saved image (higher means better quality)")
    args = parser.parse_args()

    with open(args.file1, 'r', encoding='utf-8') as file1:
        index1 = file1.readlines()
    with open(args.file2, 'r', encoding='utf-8') as file2:
        index2 = file2.readlines()
    with open(args.file3, 'r', encoding='utf-8') as file3:
        index3 = file3.readlines()

    df = pd.read_csv(args.test_file)
    api_list = df['target_api'].apply(parse_string_into_apis).tolist()
    print(api_list[:10])

    # Read the content from the files and convert to sets of integers
    with open(args.file1, 'r') as file1, open(args.file2, 'r') as file2, open(args.file3, 'r') as file3:
        set1 = set(map(int, file1.read().strip().split()))
        set2 = set(map(int, file2.read().strip().split()))
        set3 = set(map(int, file3.read().strip().split()))

    # Create a Venn diagram using the sets
    plt.figure(figsize=(8, 8))
    venn_diagram = venn3([set1, set2, set3], ('CodeBERT', 'CodeT5', 'MulaRec'))

    # Add a title and show the plot
    plt.title("Venn Diagram of Exact Match Indexes Across CodeBERT, CodeT5, and MulaRec")
    
    # Save the figure with high resolution
    plt.savefig(args.output, dpi=args.dpi)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()