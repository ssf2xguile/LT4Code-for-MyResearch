import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm
import json
import os
import logging
import argparse

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
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.classifier = nn.Linear(self.t5.config.d_model, 1)  # 二値分類用の線形層

    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(encoder_outputs.last_hidden_state[:, 0, :])  # CLSトークンに対応する出力
        logits = logits.squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())

        return {"logits": logits, "loss": loss}

class SimpleDataset(Dataset):
    """Dataset for loading and processing training data."""
    def __init__(self, file_path, tokenizer, max_length):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                js = json.loads(line.strip())
                input_text = js["func"].lstrip().lower()
                label = js["target"]
                tokenized = tokenizer(
                    input_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
                input_ids = torch.tensor(tokenized["input_ids"])
                self.examples.append((input_ids, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])  # バッチ内のinput_idsを結合
    labels = torch.tensor([item[1] for item in batch])  # ラベルをテンソル化
    return input_ids, labels

def main():
    # 設定
    model_name = "Salesforce/codet5-base"
    train_file = "../../../all_data/RQ4_data/api_data_90_headAPIMethod/train.jsonl"  # ファインチューニング用データ
    output_dir = "./saved_models_api_codet5_90_headAPIMethod"
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
    model = T5ForBinaryClassification(model_name).to(device)

    # データセットとデータローダー
    logger.info("Loading dataset...")
    train_dataset = SimpleDataset(train_file, tokenizer, max_length)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=train_batch_size,
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn,
    )
    logger.info(f"Loaded dataset with {len(train_dataset)} examples")

    # オプティマイザの準備
    logger.info("Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # ファインチューニング
    logger.info("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        logger.info(f"Epoch {epoch + 1}/{num_epochs} started")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        logger.info(f"Epoch {epoch + 1} finished with Loss: {epoch_loss / len(train_dataloader):.4f}")

    # モデルの保存
    logger.info("Saving model...")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()