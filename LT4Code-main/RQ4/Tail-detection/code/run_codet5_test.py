import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import RobertaTokenizer, T5ForConditionalGeneration  # 必要なモジュールを追加
from tqdm import tqdm
import json
import os
import logging

# ログ設定
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class T5ForBinaryClassification(nn.Module):
    """Custom T5 model for binary classification."""
    def __init__(self, model_name):
        super(T5ForBinaryClassification, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)  # 修正箇所
        self.classifier = nn.Linear(self.t5.config.d_model, 1)  # 二値分類用の線形層

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(encoder_outputs.last_hidden_state[:, 0, :])  # CLSトークンに対応する出力
        logits = logits.squeeze(-1)
        return logits

class SimpleDataset(Dataset):
    """Dataset for loading and processing test data."""
    def __init__(self, file_path, tokenizer, max_length):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                js = json.loads(line.strip())
                input_text = js["func"].lstrip().lower()
                label = js["target"]  # テストデータの正解ラベル
                tokenized = tokenizer(
                    input_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
                input_ids = torch.tensor(tokenized["input_ids"])
                idx = js["idx"] if "idx" in js else None
                self.examples.append((input_ids, label, idx))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, label, index = self.examples[idx]
        return input_ids, label, index

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])  # バッチ内のinput_idsを結合
    labels = torch.tensor([item[1] for item in batch])  # ラベルをテンソル化
    indices = [item[2] for item in batch]  # テストデータのインデックス
    return input_ids, labels, indices

def main():
    # 設定
    model_name = "Salesforce/codet5-base"
    model_path = "./saved_models_api_codet5"  # ファインチューニング済みモデルのパス
    test_file = "../../../all_data/RQ4_data/api_data/test.jsonl"  # テストデータ
    max_length = 256
    test_batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ログに設定を出力
    logger.info(f"Test file: {test_file}")
    logger.info(f"Device: {device}")

    # トークナイザーとモデルのロード
    logger.info("Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForBinaryClassification(model_name).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    model.eval()

    # データセットとデータローダー
    logger.info("Loading test dataset...")
    test_dataset = SimpleDataset(test_file, tokenizer, max_length)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=test_batch_size,
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn,
    )
    logger.info(f"Loaded test dataset with {len(test_dataset)} examples")

    # テストの実行
    logger.info("Starting testing...")
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            inputs, batch_labels, batch_indices = batch[0].to(device), batch[1].to(device), batch[2]
            logits = model(input_ids=inputs)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()  # 二値分類（0 or 1）
            predictions.extend(preds)
            labels.extend(batch_labels.cpu().numpy())

    # 正解率を計算
    correct = sum([1 for pred, label in zip(predictions, labels) if pred == label])
    total = len(labels)
    accuracy = correct / total * 100
    logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

if __name__ == "__main__":
    main()
