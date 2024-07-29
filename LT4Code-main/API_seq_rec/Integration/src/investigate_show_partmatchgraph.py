# coding=UTF-8
import matplotlib.pyplot as plt
import numpy as np

def main():
    # データ
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']
    codebert = [33.0, 11.8, 5.2, 2.6, 1.8, 1.0, 0.6, 0.6, 0.2, 1.4]
    codet5 = [31.0, 12.5, 6.1, 3.2, 2.1, 1.1, 0.7, 0.7, 0.3, 1.9]
    mularec = [30.8, 12.7, 6.4, 3.7, 2.8, 1.7, 1.2, 1.0, 0.5, 3.1]

    x = np.arange(len(labels))  # ラベルの位置
    width = 0.2  # バーの幅

    fig, ax = plt.subplots(figsize=(10, 6))  # 図の作成

    # 棒グラフの描画
    rects1 = ax.bar(x - width, codebert, width, label='CodeBERT', color='#4286f5')
    rects2 = ax.bar(x, codet5, width, label='CodeT5', color='#ea4235')
    rects3 = ax.bar(x + width, mularec, width, label='MulaRec', color='#fabd03')

    # ラベル付け
    ax.set_xlabel('Number of matching API methods')
    ax.set_ylabel('Exact Match Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # 画像を保存
    plt.savefig('../data/Exact_Match_Rate_by_Prefix_Match.png', format='png', dpi=200)
    plt.show()

if __name__ == '__main__':
    main()
