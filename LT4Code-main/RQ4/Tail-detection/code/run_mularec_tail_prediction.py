import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import pandas as pd
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

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLSトークンに対応する出力
        logits = self.classifier(pooled_output)
        return logits

class BinaryClassificationDataset(Dataset):
    """Dataset for binary classification."""
    def __init__(self, file_path, tokenizer, max_length):
        self.examples = []
        data = pd.read_csv(file_path)
        for _, row in data.iterrows():
            annotation = str(row["annotation"]).strip()
            source_code = str(row["source_code"]).strip()
            input_text = f"{annotation} {source_code}"  # 空白で結合
            tokenized = tokenizer(
                input_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            self.examples.append((input_ids, attention_mask))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def predict(model, dataloader, device, output_file):
    """Make predictions with the binary classification model."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids, attention_mask = [b.to(device) for b in batch]
            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(-1)
            predictions.extend(preds.cpu().tolist())

    # 結果をテキストファイルに保存
    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    logger.info(f"Predictions saved to {output_file}")

def main():
    # 設定
    model_name = "microsoft/codebert-base"
    test_file = "../../../all_data/api_seq_data/mularec_data/test_3_lines.csv"
    model_dir = "./saved_models_api_mularec"
    max_length = 256
    test_batch_size = 32
    output_file = "./mularec_tail_prediction.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ログに設定を出力
    logger.info(f"Using model: {model_name}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Device: {device}")

    # トークナイザーとモデルのロード
    logger.info("Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = MulaRecBinaryClassification(model_name).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))

    # データセットとデータローダー
    logger.info("Loading test dataset...")
    test_dataset = BinaryClassificationDataset(test_file, tokenizer, max_length)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=test_batch_size,
    )
    logger.info(f"Loaded test dataset with {len(test_dataset)} examples")

    # 予測
    logger.info("Starting predictions...")
    predict(model, test_dataloader, device, output_file)

if __name__ == "__main__":
    main()
