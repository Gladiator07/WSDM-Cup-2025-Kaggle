import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time

import numpy as np
import pandas as pd
import torch
from accelerate import PartialState
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import RewardTrainer, RewardConfig
from trl.trainer.utils import RewardDataCollatorWithPadding

state = PartialState()
print = state.print


def print_line():
    print("\n" + "#" + "-" * 100 + "#" + "\n")


class SequenceProcessor:
    def __init__(
        self,
        model_name: str,
        max_length: int = 1600,
        prompt_ratio: float = 0.3,  # Default 30% for prompt
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            add_prefix_space=False,
        )

        self.max_length = max_length
        self.prompt_ratio = prompt_ratio
        self.template_overhead = 10

    def truncate_if_needed(self, tokens, max_tokens):
        """Truncate tokens if they exceed max_tokens by keeping start and end portions."""
        if len(tokens) <= max_tokens:
            return tokens

        ellipsis_tokens = self.tokenizer.encode(" [...] ", add_special_tokens=False)

        keep_tokens = (max_tokens - len(ellipsis_tokens)) // 2
        return tokens[:keep_tokens] + ellipsis_tokens + tokens[-keep_tokens:]

    def process_single_sequence(self, prompt, response):
        """Process a single prompt-response pair."""
        available_tokens = self.max_length - self.template_overhead

        # Initial token allocation
        prompt_max = int(available_tokens * self.prompt_ratio)
        response_max = available_tokens - prompt_max

        # Tokenize without special tokens
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        # Calculate actual needed tokens and redistribute
        prompt_needed = min(len(prompt_tokens), prompt_max)
        excess_tokens = prompt_max - prompt_needed
        response_max = response_max + excess_tokens

        # Apply truncation if needed
        prompt_tokens = self.truncate_if_needed(prompt_tokens, prompt_needed)
        response_tokens = self.truncate_if_needed(response_tokens, response_max)

        # Decode back to text while preserving format
        prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=False)

        # Create conversation format
        conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]

        # Apply chat template
        tokenized_conversation = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
        )

        return {"input_ids": tokenized_conversation, "attention_mask": [1] * len(tokenized_conversation)}

    def process_sequence(self, row):
        """Process row with prompt and responses, handling winner selection."""
        prompt = row["prompt"]
        response_a = row["response_a"]
        response_b = row["response_b"]
        winner = row["winner"]

        # Process both sequences
        sequence_a = self.process_single_sequence(prompt, response_a)
        sequence_b = self.process_single_sequence(prompt, response_b)

        # Assign chosen/rejected based on winner
        if winner == "model_a":
            return {
                "input_ids_chosen": sequence_a["input_ids"],
                "attention_mask_chosen": sequence_a["attention_mask"],
                "input_ids_rejected": sequence_b["input_ids"],
                "attention_mask_rejected": sequence_b["attention_mask"],
                "length": max(len(sequence_a["input_ids"]), len(sequence_b["input_ids"])),
            }
        else:  # model_b
            return {
                "input_ids_chosen": sequence_b["input_ids"],
                "attention_mask_chosen": sequence_b["attention_mask"],
                "input_ids_rejected": sequence_a["input_ids"],
                "attention_mask_rejected": sequence_a["attention_mask"],
                "length": max(len(sequence_a["input_ids"]), len(sequence_b["input_ids"])),
            }


def show_sample(ds, idx, tokenizer):
    print(ds[idx])
    print("\nFormatted Text:\n")
    print("Chosen completion:\n")
    print(tokenizer.decode(ds[idx]["input_ids_chosen"]))
    print("\nRejected completion:\n")
    print(tokenizer.decode(ds[idx]["input_ids_rejected"]))
    print_line()


def main(args):
    # load test data
    test_df = pd.read_parquet("test.pq")

    print_line()
    print(f"Config:\n{args}")
    print_line()

    print(f"Total test set samples: {len(test_df)}")
    test_ds = Dataset.from_pandas(test_df)

    with state.main_process_first():
        processor = SequenceProcessor(args.base_model_path, args.max_length)

        tok_ds = test_ds.map(
            processor.process_sequence,
            batched=False,
            num_proc=4,
            remove_columns=[c for c in test_ds.column_names if c not in ["id"]],
            desc="Tokenizing test data",
        )
        # test_exceed = test_ds.filter(lambda x: x["length"] > MAX_LENGTH)

    tok_ds = tok_ds.sort("length", reverse=True)

    print_line()
    idx = 0
    show_sample(tok_ds, idx, processor.tokenizer)

    # load model
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.float16,  # only fp16 supported on T4s
        attn_implementation=args.attn_implementation,
    )

    if args.quantize_type == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
        )
    elif args.quantize_type == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=False,
        )

    state.wait_for_everyone()

    print_line()
    print(f"\nModel loading kwargs:\n{model_kwargs}\n")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        **model_kwargs,
    )
    # as some base model have only one output label
    # model.score = nn.Linear(model.config.hidden_size, 2, bias=False)
    # model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.num_labels = 1
    model.config.attn_logit_softcapping = None

    # load lora adapters
    model = PeftModel.from_pretrained(model, args.lora_path)

    print_line()
    print(f"Model Architecture:\n{model}\n")
    print_line()

    # start inference -----
    length_thresholds = [(4096, 2048), (2048, 1024), (1024, 512), (512, 0)]
    batch_sizes = [2, 8, 16, 32]

    all_logits = []
    all_ids = []
    start_time = time.perf_counter()
    for threshold, batch_size in zip(length_thresholds, batch_sizes):
        print_line()

        filtered_tok_ds = tok_ds.filter(lambda x: x["length"] <= threshold[0] and x["length"] > threshold[1])
        print(f"Threshold: {threshold}, Batch size: {batch_size}, Filtered samples: {len(filtered_tok_ds)}")

        if len(filtered_tok_ds) == 0:
            continue

        filtered_tok_ds = filtered_tok_ds.remove_columns("length")
        ids = list(filtered_tok_ds["id"])
        filtered_tok_ds = filtered_tok_ds.remove_columns("id")
        filtered_tok_ds = filtered_tok_ds.with_format("torch")

        trainer_args = RewardConfig(
            "output",
            fp16=True,
            fp16_full_eval=True,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,
            report_to="none",
            max_length=args.max_length,
        )

        trainer = RewardTrainer(
            model,
            trainer_args,
            train_dataset=filtered_tok_ds,
            eval_dataset=filtered_tok_ds,
            tokenizer=processor.tokenizer,
            data_collator=RewardDataCollatorWithPadding(processor.tokenizer),
        )

        logits = trainer.predict(filtered_tok_ds).predictions
        print(f"logits shape: {logits.shape}")
        all_logits.append(logits)
        all_ids.extend(ids)
        print_line()

    all_logits = np.concatenate(all_logits)

    print(f"all logits shape: {all_logits.shape}")
    print(f"total ids: {len(all_ids)}")

    elapsed_time = time.perf_counter() - start_time
    print(f"Total time taken for {len(test_df)} samples: {elapsed_time:.2f} seconds")
    print(f"Approx time for running 10k sample will be {(elapsed_time / len(test_df)) * 10000:.2f} seconds")

    # save logits
    logits_df = pd.DataFrame({"id": all_ids, "logits": all_logits.tolist()})
    save_path = f"{args.save_name}_logits.pq"
    logits_df.to_parquet(save_path, index=False)
    print(f"Logits saved at: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--attn_implementation", type=str)
    parser.add_argument("--quantize_type", type=str, choices=["4bit", "8bit"])
    parser.add_argument("--save_name", type=str)
    args = parser.parse_args()

    main(args)
