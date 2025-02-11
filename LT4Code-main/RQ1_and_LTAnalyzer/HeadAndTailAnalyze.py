import argparse
import json

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
def analyze_data(file1_data, file2_data, file3_dict):
    target_data = [entry for entry in file2_data if entry.get('target') == 0]
    results = []

    for entry in target_data[:100]:  # Limit to 30 entries
        idx = entry['idx']
        if idx >= len(file1_data):
            continue

        api_sequence = file1_data[idx]
        head_count, total_count = 0, 0

        for api in api_sequence:
            api_info = file3_dict.get(api)
            if api_info:
                total_count += 1
                if api_info['id'] <= 212:
                    head_count += 1

        if total_count > 0:
            head_ratio = round((head_count / total_count) * 100, 3)
            results.append((api_sequence, head_ratio))

    for seq, ratio in results:
        print(f"API Sequence: {seq}, Head Ratio: {ratio}%")

# Main function
def main():
    parser = argparse.ArgumentParser(description='Process API method data.')
    parser.add_argument('--file1', type=str, default='../all_data/api_seq_data/codet5_data/codet5_format_data/refine/small/train.buggy-fixed.fixed')
    parser.add_argument('--file2', type=str, default='../all_data/RQ4_data/api_data/train.jsonl')
    parser.add_argument('--file3', type=str, default='./api_rec_data.json')
    args = parser.parse_args()

    file1_data = process_file1(args.file1)
    file2_data = process_file2(args.file2)
    file3_dict = process_file3(args.file3)

    analyze_data(file1_data, file2_data, file3_dict)

if __name__ == '__main__':
    main()
