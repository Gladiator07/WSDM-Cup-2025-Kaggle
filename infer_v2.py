import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import simple_parsing
import torch
import torch.nn as nn
from accelerate import Accelerator, PartialState
from datasets import Dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

state = PartialState()
print = state.print


@dataclass
class Config:
    base_model_path: str = "gemma2_9b"
    lora_path: str = "none"
    max_length: int = 4096
    add_eos_token: bool = False
    attn_implementation: str = "sdpa"
    quantize_type: str = "8bit"  # 4bit
    save_name: str = "gemma2_9b"
    tta: bool = False


def print_line():
    print("\n" + "#" + "-" * 100 + "#" + "\n")


class SequenceProcessor:
    def __init__(
        self,
        model_name: str,
        max_length: int = 1600,
        truncation_side: str = "left",
        padding_side: str = "left",
        add_eos_token: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # use_fast=True,
            # trust_remote_code=True,
            # from_slow=True,
            # add_prefix_space=False,
            padding_side=padding_side,
            truncation_side=truncation_side,
        )
        self.tokenizer.add_eos_token = add_eos_token
        self.model_name = model_name

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
        buffer_tokens = 3
        available_tokens = self.max_length - self.template_length - buffer_tokens

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
            if "qwen" in self.model_name.lower():
                final_sequence.pop(0)
            if self.tokenizer.add_eos_token:
                final_sequence.append(self.tokenizer.eos_token_id)

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
        if "qwen" in self.model_name.lower():
            final_sequence.pop(0)
        if self.tokenizer.add_eos_token:
            final_sequence.append(self.tokenizer.eos_token_id)
        return {"input_ids": final_sequence, "attention_mask": [1] * len(final_sequence), "length": len(final_sequence)}


@torch.no_grad()
def get_logits(dataloader, accelerator, model):
    model.eval()
    all_logits = []
    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(dataloader):
        with accelerator.autocast():
            outputs = model(**batch)

        logits = outputs.logits
        logits = accelerator.gather_for_metrics(logits).detach().cpu().numpy()
        all_logits.extend(logits)
        progress_bar.update(1)

    progress_bar.close()

    return all_logits


def main(args):
    # load test data
    test_df = pd.read_parquet("test.pq")
    accelerator = Accelerator(mixed_precision="fp16")

    print_line()
    cfg_dict = asdict(args)
    print(f"Config:\n{cfg_dict}\n")
    print_line()

    print(f"Total test set samples: {len(test_df)}")
    print(f"TTA: {args.tta}")
    test_ds = Dataset.from_pandas(test_df)

    with state.main_process_first():
        processor = SequenceProcessor(
            args.base_model_path, args.max_length, truncation_side="left", padding_side="left", add_eos_token=args.add_eos_token
        )

        tok_ds = test_ds.map(
            processor.tokenize,
            batched=False,
            num_proc=4,
            fn_kwargs={"tta": args.tta},
            remove_columns=[c for c in test_ds.column_names if c not in ["id"]],
            desc="Tokenizing test data",
        )
        # test_exceed = test_ds.filter(lambda x: x["length"] > MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer=processor.tokenizer)

    tok_ds = tok_ds.sort("length", reverse=True)

    print_line()
    idx = 0
    print(f"\nSample tokenized data:\n{tok_ds[idx]}\n")
    print_line()

    print_line()
    print(f"{processor.tokenizer.decode(tok_ds[idx]['input_ids'])}")
    print_line()

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
    if args.lora_path != "none":
        # as some base model have only one output label
        model.score = nn.Linear(model.config.hidden_size, 2, bias=False)
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.num_labels = 2

        # load lora adapters
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.attn_logit_softcapping = None
    model.eval()
    model = accelerator.prepare(model)
    print_line()
    print(f"Model Architecture:\n{model}\n")
    print_line()

    # start inference -----
    if args.quantize_type == "4bit":
        length_thresholds = [(4096, 2048), (2048, 1024), (1024, 512), (512, 0)]
        batch_sizes = [2, 8, 16, 32]
    elif args.quantize_type == "8bit":
        length_thresholds = [(4096, 2048), (2048, 1024), (1024, 512), (512, 0)]
        batch_sizes = [1, 2, 4, 8]

    all_logits = []
    all_ids = []
    start_time = time.perf_counter()
    for threshold, batch_size in zip(length_thresholds, batch_sizes):
        print_line()

        filtered_tok_ds = tok_ds.filter(lambda x: x["length"] <= threshold[0] and x["length"] > threshold[1])
        print(f"Threshold: {threshold}, Batch size: {batch_size}, Filtered samples: {len(filtered_tok_ds)}")

        if len(filtered_tok_ds) == 0:
            continue

        ids = list(filtered_tok_ds["id"])
        keep_cols = ["input_ids", "attention_mask"]
        filtered_tok_ds = filtered_tok_ds.remove_columns(column_names=[col for col in filtered_tok_ds.column_names if col not in keep_cols])
        filtered_tok_ds = filtered_tok_ds.with_format("torch")

        dataloader = torch.utils.data.DataLoader(
            filtered_tok_ds, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=data_collator
        )
        dataloader = accelerator.prepare(dataloader)

        logits = get_logits(dataloader, accelerator, model)
        # print(f"logits shape: {logits}")
        # print(f"type logits: {type(logits)}")

        all_logits.append(logits)
        all_ids.extend(ids)
        torch.cuda.empty_cache()
        gc.collect()
        print_line()

    all_logits = np.concatenate(all_logits)
    if args.tta:
        all_logits = all_logits[:, ::-1]

    print(f"all logits shape: {all_logits.shape}")
    print(f"total ids: {len(all_ids)}")

    elapsed_time = time.perf_counter() - start_time
    print(f"Total time taken for {len(test_df)} samples: {elapsed_time:.2f} seconds")
    print(f"Approx time for running 10k sample will be {(elapsed_time / len(test_df)) * 10000:.2f} seconds")

    # save logits
    logits_df = pd.DataFrame({"id": all_ids, "logits": all_logits.tolist()})
    if args.tta:
        save_path = f"{args.save_name}_tta_logits.pq"
    else:
        save_path = f"{args.save_name}_logits.pq"
    logits_df.to_parquet(save_path, index=False)
    print(f"Logits saved at: {save_path}")


if __name__ == "__main__":
    args = simple_parsing.parse(Config)
    main(args)
