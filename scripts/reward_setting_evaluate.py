import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # for ultra-fast downloads

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention

Gemma2Attention._flash_attn_uses_top_left_mask = False
from tqdm import tqdm
from sklearn.metrics import accuracy_score

DATA_PATH = "../data/train_combined_stage_1.parquet"
MAX_LENGTH = 2048
DEBUG = False

df = pd.read_parquet(DATA_PATH)
df = df[df["split"] == "valid"].reset_index(drop=True)
df["labels"] = df["winner"].map({"model_a": 0, "model_b": 1})
if DEBUG:
    df = df.sample(n=100, random_state=42)

# Load model and tokenizer
device = "cuda:0"
model_name = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
    bnb_8bit_use_double_quant=False,
)
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
    # quantization_config=bnb_config,
)
rm_tokenizer = AutoTokenizer.from_pretrained(model_name, tokenizer_side="left")


all_preds = []
all_labels = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["prompt"]
    response_a = row["response_a"]
    response_b = row["response_b"]
    prompt_tok = rm_tokenizer(prompt, return_tensors="np", truncation=True, max_length=512, add_special_tokens=False)
    prompt = rm_tokenizer.decode(prompt_tok["input_ids"][0], skip_special_tokens=False)
    response_a_tok = rm_tokenizer(response_a, return_tensors="np", truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
    response_a = rm_tokenizer.decode(response_a_tok["input_ids"][0], skip_special_tokens=False)
    response_b_tok = rm_tokenizer(response_b, return_tensors="np", truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
    response_b = rm_tokenizer.decode(response_b_tok["input_ids"][0], skip_special_tokens=False)

    conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_a}]
    conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_b}]

    # Format and tokenize the conversations
    # If you use `tokenize=False` with `apply_chat_template` and `tokenizer()` to tokenize the conversation,
    # remeber to remove the duplicated BOS token.
    conv1_tokenized = rm_tokenizer.apply_chat_template(
        conv1,
        tokenize=True,
        return_tensors="pt",
    ).to(device)
    # .to(device)
    conv2_tokenized = rm_tokenizer.apply_chat_template(conv2, tokenize=True, return_tensors="pt").to(device)

    # Get the reward scores
    with torch.no_grad():
        score1 = rm(conv1_tokenized).logits[0][0].item()
        score2 = rm(conv2_tokenized).logits[0][0].item()
    pred = 0 if score1 > score2 else 1
    all_preds.append(pred)
    all_labels.append(row["labels"])

# Calculate accuracy

acc = accuracy_score(all_labels, all_preds)

print(f"Accuracy: {acc} on valid data of length {len(df)}")
