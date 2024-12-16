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
                # 自然言語クエリとソースコードコンテキストを融合
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

def train(model, dataloader, optimizer, device, num_epochs=5):
    """Train the binary classification model."""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader):.4f}")

def main():
    # 設定
    model_name = "microsoft/codebert-base"
    train_file = "../../../all_data/RQ4_data/api_data/train.jsonl"  # ファインチューニング用データ
    output_dir = "./saved_models_api_mularec"
    max_length = 256
    train_batch_size = 32
    num_epochs = 5
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ログに設定を出力
    logger.info(f"Using model: {model_name}")
    logger.info(f"Training file: {train_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")

    # トークナイザーとモデルのロード
    logger.info("Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = MulaRecBinaryClassification(model_name).to(device)

    # データセットとデータローダー
    logger.info("Loading dataset...")
    train_dataset = BinaryClassificationDataset(train_file, tokenizer, max_length)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=train_batch_size,
    )
    logger.info(f"Loaded dataset with {len(train_dataset)} examples")

    # オプティマイザの準備
    logger.info("Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # ファインチューニング
    logger.info("Starting training...")
    train(model, train_dataloader, optimizer, device, num_epochs)

    # モデルの保存
    logger.info("Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
