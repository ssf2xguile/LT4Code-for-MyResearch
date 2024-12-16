import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import RobertaTokenizer, RobertaModel, AdamW
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

class MulaRecBinaryClassification(nn.Module):
    """MulaRec model for binary classification."""
    def __init__(self, model_name):
        super(MulaRecBinaryClassification, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLSトークンに対応する出力
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.squeeze(-1), labels.float())
        return logits, loss

class BinaryClassificationDataset(Dataset):
    """Dataset for binary classification."""
    def __init__(self, file_path, tokenizer, max_length):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                js = json.loads(line.strip())
                input_text = js["func"].strip()
                label = js["target"]
                tokenized = tokenizer(
                    input_text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = tokenized["input_ids"].squeeze(0)
                attention_mask = tokenized["attention_mask"].squeeze(0)
                self.examples.append((input_ids, attention_mask, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def test(model, dataloader, device):
    """Evaluate the binary classification model."""
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            logits, _ = model(input_ids, attention_mask)
            predictions = (torch.sigmoid(logits) > 0.5).long().squeeze(-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy, correct, total

def main():
    # 設定
    model_name = "microsoft/codebert-base"
    test_file = "../../../all_data/RQ4_data/api_data/test.jsonl"  # テスト用データ
    model_dir = "./saved_models_api_mularec"
    max_length = 256
    test_batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ログに設定を出力
    logger.info(f"Using model: {model_name}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Device: {device}")

    # トークナイザーとモデルのロード
    logger.info("Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = MulaRecBinaryClassification(model_name).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
    model.eval()

    # データセットとデータローダー
    logger.info("Loading test dataset...")
    test_dataset = BinaryClassificationDataset(test_file, tokenizer, max_length)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=RandomSampler(test_dataset),
        batch_size=test_batch_size,
    )
    logger.info(f"Loaded test dataset with {len(test_dataset)} examples")

    # テスト
    logger.info("Starting testing...")
    accuracy, correct, total = test(model, test_dataloader, device)

    # 結果出力
    logger.info(f"Test Results - Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Correct Predictions: {correct}/{total}")

if __name__ == "__main__":
    main()
