import json
import argparse
import os
import pandas as pd

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
    return df['target_api'].str.lower()  # Exclude missing values and convert to lowercase

def join_api_methods(line):
    parts = line.split(' ')
    methods = []
    current_method = []
    for part in parts:
        if '.' in part:
            if current_method:
                current_method.append(part)
                methods.append(' '.join(current_method))
                current_method = []
            else:
                methods.append(part)
        else:
            current_method.append(part)
    return methods

def check_matching_lines(txt_lines, test_lines):
    matching_indices = []
    for i, txt_line in enumerate(txt_lines):
        txt_methods = join_api_methods(txt_line)
        for test_line in test_lines:
            test_methods = join_api_methods(test_line)
            match_found = False
            for txt_method in txt_methods:
                if txt_method in test_methods:
                    match_found = True
                    break
            if match_found:
                matching_indices.append(i + 1)  # 1-based index
                break
    return matching_indices

def write_matching_indices(file_path, indices):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index in indices:
            file.write(f"{index}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str, default='./predictions/MulaRec_predictions.txt',help="Path to the prediction.txt file")
    parser.add_argument('--test_file', type=str, default='../../all_data/api_seq_data/mularec_data/test_3_lines.csv', help="Path to the test.json file")
    parser.add_argument('--output_file', type=str, default='./result/MulaRec_lcp_index.txt', help="Path to save the output file")
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
        raise ValueError("Unsupported file extension. Only .json, .fixed, and .csv files are supported.")

    matching_indices = check_matching_lines(txt_lines, test_lines)

    write_matching_indices(args.output_file, matching_indices)

    # Output the number of matches
    total_lines = len(txt_lines)
    matched_lines = len(matching_indices)
    print(f"{matched_lines} out of {total_lines} lines matched.")
    print(f"Matching indices written to {args.output_file}")

if __name__ == "__main__":
    main()