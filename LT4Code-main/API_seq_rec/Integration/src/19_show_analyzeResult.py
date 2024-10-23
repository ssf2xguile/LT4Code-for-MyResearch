import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='../data/MyApproachCorrectness.csv', help="Path to the CSV file")
    args = parser.parse_args()

    # CSVデータを読み込む
    data = pd.read_csv(args.input_csv)

    # codebert_status, codet5_status, mularec_status がすべて 1 の行を抽出
    filtered_data = data[(data['codebert_status'] == 1) & (data['codet5_status'] == 1) & (data['mularec_status'] == 1)]

    # 結果を表示
    print(f"codebert_status, codet5_status, mularec_status がすべて 1 の行: {len(filtered_data)} 件")
    print(filtered_data)

    # final_status が 0, 1, 2 の件数をカウント
    final_status_counts = data['final_status'].value_counts()

    print("\nfinal_status の件数:")
    print(f"0: {final_status_counts.get(0, 0)} 件")
    print(f"1: {final_status_counts.get(1, 0)} 件")
    print(f"2: {final_status_counts.get(2, 0)} 件")

if __name__ == "__main__":
    main()
