import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import logging
import os

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

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(encoder_outputs.last_hidden_state[:, 0, :])  # CLSトークンに対応する出力
        logits = logits.squeeze(-1)
        return logits

class BuggyDataset(Dataset):
    """Dataset for processing test data from test.buggy-fixed.buggy."""
    def __init__(self, file_path, tokenizer, max_length):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:  # 空行をスキップ
                    continue

                # 前処理：`text:` と `code:` を除去
                if "text:" in line and "code:" in line:
                    text_start = line.index("text:") + len("text:")
                    code_start = line.index("code:")
                    text = line[text_start:code_start].strip()
                    code = line[code_start + len("code:"):].strip()

                    # 入力テキストを作成
                    input_text = f"{text} {code}"

                    # トークナイズとテンソル変換
                    tokenized = tokenizer(
                        input_text,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                    )
                    input_ids = torch.tensor(tokenized["input_ids"])
                    self.examples.append((input_ids, input_text))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])  # バッチ内のinput_idsを結合
    input_texts = [item[1] for item in batch]  # 生テキストを保持
    return input_ids, input_texts

def main():
    # 設定
    model_name = "Salesforce/codet5-base"
    model_path = "./saved_models_api_codet5"  # ファインチューニング済みモデルのパス
    test_file = "../../../all_data/api_seq_data/codet5_data/codet5_format_data/refine/small/test.buggy-fixed.buggy"  # テストデータ
    output_file = "./codet5_tail_prediction.txt"  # 結果を保存するファイル
    max_length = 256
    test_batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ログに設定を出力
    logger.info(f"Using model: {model_name}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Device: {device}")

    # トークナイザーとモデルのロード
    logger.info("Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForBinaryClassification(model_name).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    model.eval()

    # データセットとデータローダー
    logger.info("Loading test dataset...")
    test_dataset = BuggyDataset(test_file, tokenizer, max_length)
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
    input_texts = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            inputs, batch_texts = batch[0].to(device), batch[1]
            logits = model(input_ids=inputs)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()  # 二値分類（0 or 1）
            predictions.extend(preds)
            input_texts.extend(batch_texts)

    print(input_texts[:5])

    # 結果をファイルに保存
    logger.info("Saving predictions...")
    with open(output_file, "w") as f:
        for input_text, pred in zip(input_texts, predictions):
            f.write(f"{pred}\n")
    logger.info(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
