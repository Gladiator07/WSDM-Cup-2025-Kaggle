import gc
import os
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # for ultra-fast downloads

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from huggingface_hub import HfApi
from sequence_processor import SequenceProcessor
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed, BitsAndBytesConfig
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from peft import PeftModel

Gemma2Attention._flash_attn_uses_top_left_mask = False

from dotenv import load_dotenv

load_dotenv()
hf_api = HfApi()


class Config:
    # Paths
    STAGE1_DATA_PATH = "data/train_combined_stage_1_v3.parquet"
    DATA_PATH = "data/pseudo_labeled_final_data/train_stage2_stage1_combined_v1.parquet"
    OUT_DIR = "./artifacts"
    HF_DATA_REPO_ID = "Gladiator/wsdm-cup-datasets"
    FOLDER_IN_REPO = "pseudo_labeled_data_final"

    # Model config
    # MODEL_ID = "Gladiator/wsdm_cup_gemma2_27B_stage1_lora_707_merged"
    # MODEL_ID = "Gladiator/wsdm_cup_gemma2_9b_stage1_init_stage0_merged"
    MODEL_ID = "Gladiator/wsdm_cup_phi4_stage1_lora_700_merged"
    ADD_EOS_TOKEN = True  # Important check this (should be true for phi4)
    GEMMA_DISABLE_ATTN_LOGIT_SOFTCAP = False  # important check this (should be false for gemma 27b)
    # GEMMA_DISABLE_ATTN_LOGIT_SOFTCAP = True
    SAVE_NAME = MODEL_ID.split("/")[-1].replace("wsdm_cup_", "pl_")

    # Misc config
    DEBUG = False
    PSEUDO_LABEL_EVAL_DATA = False
    MAX_LENGTH = 4096
    LENGTH_THRESHOLDS = [(4096, 2048), (2048, 1024), (1024, 512), (512, 0)]
    # BATCH_SIZES = [8, 16, 32, 64]
    # BATCH_SIZES = [16, 32, 64, 128]
    # BATCH_SIZES = [32, 64, 128, 256]
    BATCH_SIZES = [64, 128, 256, 512]
    QUANTIZE_TO_4BIT = True


cfg = Config()


def process_both_logits():
    start_time = time.time()
    set_seed(2025)

    config_dict = {x: dict(Config.__dict__)[x] for x in dict(Config.__dict__) if not x.startswith("_")}
    print(f"Config: {config_dict}")

    # Load data
    if cfg.PSEUDO_LABEL_EVAL_DATA:
        df = pd.read_parquet(cfg.STAGE1_DATA_PATH)
        df = df[df["split"] == "valid"].reset_index(drop=True)
    else:
        df = pd.read_parquet(cfg.DATA_PATH)

    if cfg.DEBUG:
        df = df.sample(1000, random_state=42)

    print(f"Loaded dataset with {len(df)} samples")

    print(f"Sources: {df.source.value_counts().to_dict()}")

    # Create datasets for both regular and TTA
    processor = SequenceProcessor(
        cfg.MODEL_ID, max_length=cfg.MAX_LENGTH, truncation_side="left", padding_side="left", add_eos_token=cfg.ADD_EOS_TOKEN
    )

    # Initialize base dataset
    ds = Dataset.from_pandas(df)

    # Process for both regular and TTA in one go
    def process_both(example):
        regular_result = processor.tokenize(example, tta=False)
        tta_result = processor.tokenize(example, tta=True)
        # Combine the results
        return {
            "input_ids": regular_result["input_ids"],
            "attention_mask": regular_result["attention_mask"],
            "length": regular_result["length"],
            "input_ids_tta": tta_result["input_ids"],
            "attention_mask_tta": tta_result["attention_mask"],
        }

    ds = ds.map(process_both, batched=False, num_proc=8, desc="Tokenizing data for both regular and TTA")
    ds = ds.sort("length", reverse=True)  # sort by descending so OOM occurs earlier

    quant_config = None
    if cfg.QUANTIZE_TO_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cfg.model.dtype
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    if cfg.GEMMA_DISABLE_ATTN_LOGIT_SOFTCAP:
        model.config.attn_logit_softcapping = None

    print(model.config)
    print(model)

    all_processed_ds = []
    for threshold, batch_size in zip(cfg.LENGTH_THRESHOLDS, cfg.BATCH_SIZES):
        print("-" * 80)
        filtered_ds = ds.filter(lambda x: x["length"] <= threshold[0] and x["length"] > threshold[1], num_proc=8)

        if len(filtered_ds) == 0:
            continue

        print(f"Threshold: {threshold}, batch_size: {batch_size}, total samples: {len(filtered_ds)}, inference with TTA=False")

        # Process regular logits
        training_args = TrainingArguments(
            output_dir="./tmp", per_device_eval_batch_size=batch_size, report_to="none", torch_compile=True, bf16_full_eval=True
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorWithPadding(processor.tokenizer),
        )

        # For regular prediction
        regular_ds = filtered_ds.remove_columns(["input_ids_tta", "attention_mask_tta"])

        regular_out = trainer.predict(test_dataset=regular_ds)
        regular_logits = regular_out.predictions

        # For TTA prediction
        tta_ds = filtered_ds.remove_columns(["input_ids", "attention_mask"]).rename_columns(
            {"input_ids_tta": "input_ids", "attention_mask_tta": "attention_mask"}
        )

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Threshold: {threshold}, batch_size: {batch_size}, total samples: {len(filtered_ds)}, inference with TTA=True")
        tta_out = trainer.predict(test_dataset=tta_ds)
        tta_logits = tta_out.predictions[:, ::-1]  # reverse logits for TTA

        # Add both logits to the dataset
        keep_cols = df.columns.tolist() + ["logits", "logits_tta"]
        result_ds = filtered_ds.remove_columns([col for col in filtered_ds.column_names if col not in keep_cols])
        result_ds = result_ds.add_column("logits", [logit.tolist() for logit in regular_logits])
        result_ds = result_ds.add_column("logits_tta", [logit.tolist() for logit in tta_logits])

        all_processed_ds.append(result_ds)

        print("-" * 80)
        torch.cuda.empty_cache()
        gc.collect()

    # Combine all processed datasets
    save_name = f"{cfg.SAVE_NAME}.parquet"
    final_ds = concatenate_datasets(all_processed_ds)
    output_path = os.path.join(cfg.OUT_DIR, save_name)
    final_df = final_ds.to_pandas()
    print(f"final dataset shape: {final_df.shape}")
    final_df.to_parquet(output_path)

    # Upload to HuggingFace
    hf_api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=f"{cfg.FOLDER_IN_REPO}/{save_name}",
        repo_id=cfg.HF_DATA_REPO_ID,
        repo_type="model",
    )

    elapsed_time = time.time() - start_time
    print(f"Finished processing {len(final_df)} samples in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    process_both_logits()
