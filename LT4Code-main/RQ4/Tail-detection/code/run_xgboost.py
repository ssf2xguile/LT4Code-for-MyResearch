import os
import argparse
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_filename", default=None, type=str,
                        help="An optional input model file to load")

    args = parser.parse_args()

    # ディレクトリが存在しなければ作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # データの読み込み
    train = pd.read_json(args.train_data_file, lines=True, orient="records")
    valid = pd.read_json(args.eval_data_file, lines=True, orient="records")
    test = pd.read_json(args.test_data_file, lines=True, orient="records")

    X_train = train["func"]
    y_train = train["target"]
    X_valid = valid["func"]
    y_valid = valid["target"]
    X_test = test["func"]
    y_test = test["target"]
    logger.info("Data load completed")

    # テキストの前処理とベクトル化
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)
    X_test_vec = vectorizer.transform(X_test)
    logger.info("Vectorization completed")

    # モデルの読み込み or 新規作成
    model_path = os.path.join(args.output_dir, args.model_filename)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded existing model.")
    else:
        # GPU使用設定
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.1,
            'verbosity': 1,  # 学習過程を出力
            'device': 'cuda', # CUDAを使用
            'tree_method': 'hist',  # ヒストグラムをベースにした方法
        }

        # データセットの作成
        dtrain = xgb.DMatrix(X_train_vec, label=y_train)
        dvalid = xgb.DMatrix(X_valid_vec, label=y_valid)

        # モデルの学習
        evals = [(dtrain, 'train'), (dvalid, 'valid')]
        model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=50)

        # モデルの保存
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("Trained and saved new model.")

    # モデル学習時に早期停止を使用して最適な木の数を決定することを想定します。
    # 'bst' が訓練済みモデルで、'early_stopping_rounds'が学習時に使用されているとします。
    # Pythonではインデックスが0から始まるので、1を足して調整します。
    best_ntree_limit = model.best_iteration + 1

    dtest = xgb.DMatrix(X_test_vec)

    # テストデータでの評価
    y_pred = model.predict(dtest, iteration_range=(0, best_ntree_limit))
    y_pred = [1 if x > 0.5 else 0 for x in y_pred]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

if __name__ == "__main__":
    main()
