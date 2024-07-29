import os
import argparse
import pickle
import pandas as pd
import lightgbm as lgb
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

    # データの準備
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
    logger.info("Vectorize completed")

    # モデルの読み込み or 新規作成
    model_path = os.path.join(args.output_dir, args.model_filename)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded existing model.")
    else:
        # ハイパーパラメータの設定
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': 5,
            'num_leaves': 32,
            'learning_rate': 0.1,
            'verbose': 1  # 学習過程を出力
        }

        # データセットの作成
        dtrain = lgb.Dataset(X_train_vec, label=y_train)
        dvalid = lgb.Dataset(X_valid_vec, label=y_valid)

        # モデルの学習
        model = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dtrain, dvalid], callbacks=[lgb.early_stopping(50)])

        # モデルの保存
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("Trained and saved new model.")

    # テストデータでの評価
    y_pred = model.predict(X_test_vec, num_iteration=model.best_iteration)
    y_pred = [1 if x > 0.5 else 0 for x in y_pred]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

if __name__ == "__main__":
    main()