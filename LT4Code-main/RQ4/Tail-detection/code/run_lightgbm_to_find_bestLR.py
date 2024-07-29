import os
import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

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

    # ハイパーパラメータの設定
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "max_depth": 5,
        "num_leaves": 32,
        "seed": 42,
        "deterministic": True,
        "verbose": -1,
    }

    # 学習率を変化させながら精度と最適なイテレーション数を記録する
    best_iterations = {}
    best_scores = {}
    for lr in [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]:
        lgb_params["learning_rate"] = lr
        callbacks = [
            lgb.log_evaluation(period=100, show_stdv=True),
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True),
        ]

        dtrain = lgb.Dataset(X_train_vec, label=y_train)
        dvalid = lgb.Dataset(X_valid_vec, label=y_valid)

        cv_result = lgb.cv(
            params=lgb_params,
            train_set=dtrain,
            num_boost_round=1000,
            callbacks=callbacks,
            return_cvbooster=True,
        )

        best_iteration = cv_result["cvbooster"].best_iteration
        best_iterations[lr] = best_iteration
        logger.info(f"Best iteration (lr: {lr:.4f}): {best_iteration}")

        # テストデータでの評価
        model = cv_result["cvbooster"].boosters[0]
        y_pred = model.predict(X_test_vec, num_iteration=best_iteration)
        y_pred = [1 if x > 0.5 else 0 for x in y_pred]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        logger.info(f"Test Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

        best_scores[lr] = acc  # ここではAccuracyを使用

    # 結果を 2 軸グラフで可視化する
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(list(best_iterations.keys()), list(best_iterations.values()), marker="o", linestyle="-", color="r", label="Iterations")
    ax1.set_xlabel("Learning Rate")
    ax1.set_ylabel("Best Iteration")
    ax1.set_title("LightGBM Performance on Test Data")
    ax2 = ax1.twinx()
    ax2.plot(list(best_scores.keys()), list(best_scores.values()), marker="+", linestyle="-", color="b", label="Test Accuracy")
    ax2.set_ylabel("Test Accuracy")

    axes = [ax1, ax2]
    lines = [ax.get_legend_handles_labels()[0] for ax in axes]
    labels = [ax.get_legend_handles_labels()[1] for ax in axes]
    ax1.legend(sum(lines, []), sum(labels, []), loc="upper left")
    fig.savefig("lightgbm_test_performance.png")

    logger.info("Best iterations: %s", best_iterations)
    logger.info("Best scores: %s", best_scores)


if __name__ == "__main__":
    main()