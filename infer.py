import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from torch.amp import autocast
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def sort_and_batch_samples(data, tokenizer, max_length):
    encoded = tokenizer(data["text"].tolist(), truncation=True, max_length=max_length, padding=False, return_tensors=None)

    lengths = [len(ids) for ids in encoded["input_ids"]]
    # sort in descending order of lengths
    sorted_indices = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    # reorder data, encoded and lengths based on sorted indices
    sorted_data = data.iloc[sorted_indices].reset_index(drop=True)
    sorted_encoded = {k: [v[i] for i in sorted_indices] for k, v in encoded.items()}
    sorted_lengths = [lengths[i] for i in sorted_indices]
    # create a mapping from original index to new index
    position_map = {i: idx for idx, i in enumerate(sorted_indices)}
    return sorted_data, sorted_encoded, sorted_lengths, position_map


def dynamic_batch(sorted_data, sorted_lengths, max_tokens, tokenizer, device):
    # batches = []
    current_batch = []
    current_length = 0

    for i, length in enumerate(sorted_lengths):
        if current_length + length > max_tokens and current_batch:
            yield (
                tokenizer(
                    sorted_data.loc[current_batch, "text"].tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max(sorted_lengths[j] for j in current_batch),
                ).to(device),
                current_batch,
            )
            current_batch = []
            current_length = 0

        current_batch.append(i)
        current_length += length

    if current_batch:
        yield (
            tokenizer(
                sorted_data.loc[current_batch, "text"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(sorted_lengths[j] for j in current_batch),
            ).to(device),
            current_batch,
        )


@torch.no_grad()
def process_batches(model, batches, position_map, device):
    all_logits = []
    all_indices = []
    # consume the batches generator and load the tokenized data in memory
    # batches_list = list(batches)
    for batch, indices in tqdm(batches, desc="Processing batches"):
        with autocast(device):
            out = model(**batch)
            logits = out.logits.to(device)
            logits = F.softmax(logits, dim=1).cpu()
        all_logits.append(logits)
        all_indices.extend(indices)

    combined_logits = torch.cat(all_logits).numpy()
    reordered_logits = np.zeros_like(combined_logits)
    for i, idx in enumerate(all_indices):
        reordered_logits[position_map[idx]] = combined_logits[i]

    return reordered_logits


def split_data_by_length(data, tokenizer, max_length, num_gpus):
    """Assign sequences to GPUs based on their lengths and distribute them in a way that each GPU gets almost equal number of tokens"""
    # tokenize all text
    encoded = tokenizer(data["text"].tolist(), truncation=True, max_length=max_length, padding=False, return_tensors=None)
    # get lengths of all sequences
    lengths = [len(ids) for ids in encoded["input_ids"]]

    # sort indices by length in descending order
    # so that longer sequences are processed first, helps to see OOM if any earlier
    sorted_indices = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    # total_length = sum(lengths)
    # target_length_per_gpu = total_length // num_gpus

    # create a list of lists to store indices of sequences for each gpu
    # example: [[gpu0_indices], [gpu1_indices]]
    gpu_splits = [[] for _ in range(num_gpus)]
    # to maintain running total of lengths of sequences assigned to each gpu
    current_lengths = [0] * num_gpus

    for idx in sorted_indices:
        # find gpu with minimum total length of sequences assigned
        target_gpu = min(range(num_gpus), key=lambda i: current_lengths[i])
        # assign sequence to that gpu
        gpu_splits[target_gpu].append(idx)
        # update total length of sequences assigned to that gpu
        current_lengths[target_gpu] += lengths[idx]

    print(f"Total tokens per GPU: {current_lengths}")
    return gpu_splits


def main(args):
    print(f"Running models with following configuration: {args}")
    test = pd.read_parquet("test.pq")
    infer_mode = args.infer_mode
    model_name = args.model
    device_map = {"": args.device} if args.device != "cuda" else "auto"

    model_kwargs = dict(torch_dtype=torch.float16, device_map=device_map, trust_remote_code=True, low_cpu_mem_usage=True)

    if infer_mode == "4bit":
        print("Using 4-bit quantization")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
        )
    elif infer_mode == "8bit":
        print("Using 8-bit quantization")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16, bnb_8bit_use_double_quant=False
        )

    elif infer_mode == "na":
        print("Model is already quantized, loading as is")

    print(f"Model kwargs: {model_kwargs}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    model.config.attn_logit_softcapping = None

    model.score = nn.Linear(model.config.hidden_size, 2, bias=False).to(model.device)
    model.config.num_labels = 2

    model = PeftModel.from_pretrained(model, args.lora_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True, from_slow=True, add_prefix_space=False, padding_side="left", truncation_side="left"
    )
    tokenizer.add_eos_token = False
    model.config.pad_token_id = tokenizer.pad_token_id

    model_device = args.device
    if args.device == "cuda":
        model_device = "cuda:0"

    max_length = args.max_length

    num_gpus = 1 if args.device == "cuda" else 2
    # get indices of sequences assigned to each gpu
    gpu_splits = split_data_by_length(test, tokenizer, max_length, num_gpus)

    # print(f"Sequence assigned to each GPU: {gpu_splits}")

    orig_len = len(test)

    if args.device.startswith("cuda:"):
        # get device id (0 or 1)
        device_id = int(args.device.split(":")[1])
        # process only the sequences assigned to that GPU
        current_indices = gpu_splits[device_id]
        # get only the gpu assigned sequences
        test = test.iloc[current_indices].copy()
    elif args.device == "cuda":
        # If using all GPUs, process the entire dataset
        current_indices = gpu_splits[0]
        test = test.iloc[current_indices].copy()

    sorted_data, sorted_encoded, sorted_lengths, position_map = sort_and_batch_samples(test, tokenizer, max_length)

    max_tokens = 2048 * 2  # max tokens per batch
    # returns encoded, indices
    batches = dynamic_batch(sorted_data, sorted_lengths, max_tokens, tokenizer, args.device)

    all_logits = process_batches(model, batches, position_map, model_device)
    print(f"Logits shape: {all_logits.shape}")

    # Reorder logits to match the original data order
    final_logits = np.zeros((orig_len, all_logits.shape[1]))
    for i, orig_idx in enumerate(current_indices):
        final_logits[orig_idx] = all_logits[i]

    save_suffix = args.save_suffix
    if args.device == "cuda":
        np.save(f"logits_{save_suffix}", final_logits)
    elif args.device.startswith("cuda:"):
        device_id = args.device.split(":")[1]
        np.save(f"logits_v{device_id}_{save_suffix}", final_logits)
    else:
        np.save(f"logits_cpu_{save_suffix}", final_logits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--infer_mode", type=str, choices=["4bit", "8bit", "na"], required=True)
    parser.add_argument("--save_suffix", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    main(args)
