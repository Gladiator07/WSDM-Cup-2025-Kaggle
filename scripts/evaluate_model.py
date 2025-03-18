import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # for ultra-fast downloads

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from peft import PeftModel

Gemma2Attention._flash_attn_uses_top_left_mask = False
load_dotenv()

MODEL_ID = "sfairXC/FsfairX-Gemma2-RM-v0.1"
PEFT_MODEL_ID = "Gladiator/wsdm-cup_gemma2_9b_stage2_v1"  # None
DATA_PATH = "./data/train_combined_stage_1.parquet"
MAX_LENGTH = 4096
BATCH_SIZE = 8
DEBUG = False


class MetricCalculator:
    def __init__(self, val_df, output_dir):
        self.val_df = val_df
        self.output_dir = output_dir

    def __call__(self, eval_preds):
        # avoid modifying the class variable
        val_df = self.val_df.copy()
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        probs = torch.from_numpy(preds).float().softmax(-1).numpy()
        preds = probs.argmax(-1)
        acc = accuracy_score(y_true=labels, y_pred=preds)
        val_df["prob_0"] = probs[:, 0]
        val_df["prob_1"] = probs[:, 1]
        val_df["prediction"] = preds
        val_df.to_parquet(os.path.join(self.output_dir, "oof_preds.parquet"))
        return {"accuracy": acc}


class SequenceProcessor:
    def __init__(
        self,
        model_name: str,
        max_length: int = 1600,
        truncation_side: str = "left",
        padding_side: str = "left",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # use_fast=True,
            trust_remote_code=True,
            # from_slow=True,
            add_prefix_space=False,
            padding_side=padding_side,
            truncation_side=truncation_side,
        )
        self.tokenizer.add_eos_token = False

        self.max_length = max_length

        # Template parts (only tokenize once during initialization)
        self.templates = {
            "start": self.tokenizer.encode("# Prompt\n", add_special_tokens=False),
            "response_a": self.tokenizer.encode("\n\n# Response A\n", add_special_tokens=False),
            "response_b": self.tokenizer.encode("\n\n# Response B\n", add_special_tokens=False),
            "question": self.tokenizer.encode("\n\n# Which response is better?", add_special_tokens=False),
            "ellipsis": self.tokenizer.encode(" [...] ", add_special_tokens=False),
        }

        # Calculate fixed template length
        self.template_length = sum(len(tokens) for tokens in self.templates.values()) - len(self.templates["ellipsis"])

    def truncate_if_needed(self, tokens, max_tokens):
        """Truncate tokens if they exceed max_tokens by keeping start and end portions."""
        if len(tokens) <= max_tokens:
            return tokens

        keep_tokens = (max_tokens - len(self.templates["ellipsis"])) // 2
        return tokens[:keep_tokens] + self.templates["ellipsis"] + tokens[-keep_tokens:]

    def tokenize(self, row, tta=False):
        if tta:
            prompt, response_a, response_b = row["prompt"], row["response_b"], row["response_a"]
        else:
            prompt, response_a, response_b = row["prompt"], row["response_a"], row["response_b"]

        # Available tokens after accounting for template and special tokens
        available_tokens = self.max_length - self.template_length - 1  # -1 for BOS token

        # Tokenize all inputs at once
        enc = self.tokenizer([prompt, response_a, response_b], add_special_tokens=False)["input_ids"]
        prompt_tokens, response_a_tokens, response_b_tokens = enc[0], enc[1], enc[2]

        total_length = len(prompt_tokens) + len(response_a_tokens) + len(response_b_tokens)

        # If total length is within limit, return without truncation
        if total_length <= available_tokens:
            final_sequence = (
                [self.tokenizer.bos_token_id]
                + self.templates["start"]
                + prompt_tokens
                + self.templates["response_a"]
                + response_a_tokens
                + self.templates["response_b"]
                + response_b_tokens
                + self.templates["question"]
            )

            return {"input_ids": final_sequence, "attention_mask": [1] * len(final_sequence), "length": len(final_sequence)}

        # Allocate tokens based on 20-40-40 split with dynamic adjustment
        prompt_max = int(available_tokens * 0.2)  # Reserve 20% for prompt
        response_max = int(available_tokens * 0.4)  # 40% each for responses

        # If prompt needs less than its allocation, distribute the excess
        prompt_needed = min(len(prompt_tokens), prompt_max)
        excess_tokens = prompt_max - prompt_needed

        # Add half of excess to each response's budget
        response_a_max = response_max + excess_tokens // 2
        response_b_max = response_max + excess_tokens - (excess_tokens // 2)  # Account for odd number

        # Calculate actual token allocations
        prompt_max_tokens = prompt_needed
        response_a_max_tokens = min(len(response_a_tokens), response_a_max)
        response_b_max_tokens = min(len(response_b_tokens), response_b_max)

        # Truncate each section if needed
        prompt_tokens = self.truncate_if_needed(prompt_tokens, prompt_max_tokens)
        response_a_tokens = self.truncate_if_needed(response_a_tokens, response_a_max_tokens)
        response_b_tokens = self.truncate_if_needed(response_b_tokens, response_b_max_tokens)

        # Assemble final input
        final_sequence = (
            [self.tokenizer.bos_token_id]
            + self.templates["start"]
            + prompt_tokens
            + self.templates["response_a"]
            + response_a_tokens
            + self.templates["response_b"]
            + response_b_tokens
            + self.templates["question"]
        )
        return {"input_ids": final_sequence, "attention_mask": [1] * len(final_sequence), "length": len(final_sequence)}


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PATH)
    df["labels"] = df["winner"].map({"model_a": 0, "model_b": 1})
    eval_df = df[df["split"] == "valid"].copy().reset_index(drop=True)

    if DEBUG:
        eval_df = eval_df.sample(n=1000, random_state=42)

    eval_ds = Dataset.from_pandas(eval_df)
    processor = SequenceProcessor(MODEL_ID, max_length=MAX_LENGTH, truncation_side="left", padding_side="left")

    eval_ds = eval_ds.map(processor.tokenize, batched=False, num_proc=8, fn_kwargs={"tta": False}, desc="Tokenizing eval data")
    eval_ds = eval_ds.sort("length")

    eval_exceed = eval_ds.filter(lambda x: x["length"] > MAX_LENGTH)
    print(f"Found {len(eval_exceed)} examples exceeding max_length")

    print(f"Sample data:\n{eval_ds[0]}")

    print(f"Formatted text:\n{processor.tokenizer.decode(eval_ds[0]['input_ids'])}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2"
    )
    if PEFT_MODEL_ID is not None:
        model.score = nn.Linear(model.config.hidden_size, 2, bias=False)
        model = PeftModel.from_pretrained(model, PEFT_MODEL_ID)
        model.config.num_labels = 2

    model.to("cuda")
    model.score.to("cuda")
    model.eval()

    if "gemma" in MODEL_ID:
        model.config.attn_logit_softcapping = None

    print(f"Model architecture:\n{model}")

    training_args = TrainingArguments(
        output_dir="./tmp", per_device_eval_batch_size=BATCH_SIZE, report_to="none", torch_compile=True, bf16_full_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(processor.tokenizer),
        compute_metrics=MetricCalculator(eval_df, "./tmp"),
    )

    out = trainer.evaluate(eval_dataset=eval_ds)
    print(out)

    try:
        print(out.metrics)
    except Exception as _:
        print("haha")
