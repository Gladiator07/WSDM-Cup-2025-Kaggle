import os
import sys
import time
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # for ultra-fast downloads

import shutil

import huggingface_hub as hf_hub
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import wandb
from accelerate.state import PartialState
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from trl import RewardConfig, RewardTrainer

Gemma2Attention._flash_attn_uses_top_left_mask = False
transformers.logging.set_verbosity_info()

try:
    from discordwebhook import Discord

    notify_discord = True
except ImportError:
    notify_discord = False

load_dotenv()
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
state = PartialState()
print = state.print
# ------------------------- Helper Functions -------------------------


def print_line():
    print("\n" + "#" + "-" * 100 + "#" + "\n")


def asHours(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:.0f}h:{m:.0f}m:{s:.0f}s"


def upload_artifacts_to_hf_hub(upload_dir: str, repo_id_prefix: str, experiment_name: str, path_in_repo: str):
    hf_hub.login(token=os.environ["HF_TOKEN"], write_permission=True)
    api = hf_hub.HfApi()
    repo_id = f"{os.environ['HF_USERNAME']}/{repo_id_prefix}_{experiment_name}"
    repo_url = hf_hub.create_repo(repo_id, exist_ok=True, private=True)
    api.upload_folder(folder_path=upload_dir, repo_id=repo_id, path_in_repo=path_in_repo)
    return repo_url


# ------------------------- Data Functions -------------------------


def show_sample(ds, idx, tokenizer):
    print(ds[idx])
    print("\nFormatted Text:\n")
    print("Chosen completion:\n")
    print(tokenizer.decode(ds[idx]["input_ids_chosen"]))
    print("\nRejected completion:\n")
    print(tokenizer.decode(ds[idx]["input_ids_rejected"]))
    print_line()


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


# ------------------------- Optimizer -------------------------
def get_optimizer(model, optimizer_name, head_lr, lr, weight_decay, print_fn=None):
    no_decay = ["bias", "LayerNorm.weight"]
    head_layer_name = "score"

    # start with all of the candidate parameters
    param_dict = {name: param for name, param in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}

    # head & body params
    param_dict_head = {name: param for name, param in param_dict.items() if head_layer_name in name}
    param_dict_body = {name: param for name, param in param_dict.items() if head_layer_name not in name}

    # create groups ---
    head_params_no_decay = [param for name, param in param_dict_head.items() if any(nd in name for nd in no_decay)]
    head_params_decay = [param for name, param in param_dict_head.items() if not any(nd in name for nd in no_decay)]

    body_params_no_decay = [param for name, param in param_dict_body.items() if any(nd in name for nd in no_decay)]
    body_params_decay = [param for name, param in param_dict_body.items() if not any(nd in name for nd in no_decay)]

    optim_groups = [
        {"params": head_params_no_decay, "lr": head_lr, "weight_decay": 0},
        {"params": head_params_decay, "lr": head_lr, "weight_decay": weight_decay},
        {"params": body_params_no_decay, "lr": lr, "weight_decay": 0},
        {"params": body_params_decay, "lr": lr, "weight_decay": weight_decay},
    ]

    n_head_params_no_decay = round(sum(p.numel() for p in head_params_no_decay))
    n_head_params_decay = round(sum(p.numel() for p in head_params_decay))
    n_body_params_no_decay = round(sum(p.numel() for p in body_params_no_decay))
    n_body_params_decay = round(sum(p.numel() for p in body_params_decay))

    print_fn(f"n_head_params_no_decay: {n_head_params_no_decay:,}")
    print_fn(f"n_head_params_decay: {n_head_params_decay:,}")
    print_fn(f"n_body_params_no_decay: {n_body_params_no_decay:,}")
    print_fn(f"n_body_params_decay: {n_body_params_decay:,}")

    eight_bit_names = ["Adam8bit", "AdamW8bit", "PagedAdam8bit", "PagedAdamW8bit"]

    assert optimizer_name in eight_bit_names, f"Optimizer {optimizer_name} not supported for 8-bit optimization"

    import bitsandbytes  # type: ignore

    optimizer_cls = getattr(bitsandbytes.optim, optimizer_name)
    optimizer = optimizer_cls(
        optim_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            manager.register_module_override(module, "weight", {"optim_bits": 32})
    return optimizer


# ------------------------- Main -------------------------
def main(cfg: DictConfig):
    exp_start_time = time.time()
    out_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    main_process = state.is_main_process
    num_proc = 16

    if notify_discord and main_process:
        discord = Discord(url=os.getenv("DISCORD_WEBHOOK"))

    if cfg.seed == -1:
        cfg.seed = np.random.randint(0, 1000)
    set_seed(cfg.seed)

    if cfg.debug:
        cfg.sample_data = True
        cfg.use_wandb = False
        cfg.upload_to_hf_hub = False
        cfg.trainer_args.num_train_epochs = 1
        cfg.trainer_args.report_to = "none"

    if cfg.use_wandb and main_process:
        # delete trainer_args from cfg_to_log
        cfg_dict.pop("trainer_args")
        wandb.init(project=cfg.wandb_project_name, name=cfg.experiment_name, notes=cfg.notes, config=cfg_dict)

    # load data
    df = pd.read_parquet(os.path.join(cfg.data_dir, cfg.train_file_name))
    print(f"Data loaded with {len(df)} samples")
    if cfg.sample_data:
        df = df.sample(n=100, random_state=42).reset_index(drop=True)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "valid"].copy().reset_index(drop=True)

    start_msg = f"🚀 Starting experiment {cfg.experiment_name}"
    if notify_discord and main_process:
        discord.post(content=start_msg)

    print_line()
    print(start_msg)
    print("Configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))
    print_line()

    print_line()
    print(f"Data sources: {df['source'].value_counts().to_dict()}")
    print(f"Train Data: {len(train_df)} samples | Eval Data: {len(val_df)} samples")
    print_line()

    # tokenize data
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(val_df)
    processor = SequenceProcessor(cfg.model.model_name, max_length=cfg.tokenizer.max_length)

    tokenizer = processor.tokenizer
    with state.main_process_first():
        train_ds = train_ds.map(
            processor.process_sequence,
            batched=False,
            num_proc=num_proc,
            desc="Tokenizing train data",
        )
        eval_ds = eval_ds.map(
            processor.process_sequence,
            batched=False,
            num_proc=num_proc,
            desc="Tokenizing eval data",
        )

    # check how many of the samples exceed the max_length
    print_line()
    print("Checking how many samples exceed the max_length")
    train_exceed = train_ds.filter(lambda x: x["length"] > cfg.tokenizer.max_length, num_proc=num_proc)
    eval_exceed = eval_ds.filter(lambda x: x["length"] > cfg.tokenizer.max_length, num_proc=num_proc)
    print(f"Train samples exceeding max_length: {len(train_exceed)}")
    print(f"Eval samples exceeding max_length: {len(eval_exceed)}")
    print_line()

    # check sample with highest length to see if truncation is applied properly
    print_line()
    print(f"Train Data: {len(train_ds)} samples | Eval Data: {len(eval_ds)} samples")
    print("\nSample with highest length:\n")
    train_max_len_idx = int(np.argmax(train_ds["length"]))
    show_sample(train_ds, train_max_len_idx, tokenizer)

    # print first sample
    print("\nSample Data:")
    show_sample(train_ds, 0, tokenizer)

    # print random sample
    print_line()
    rand_idx = np.random.randint(0, len(train_ds))
    show_sample(train_ds, rand_idx, tokenizer)

    print_line()
    print("processed data:")
    print(train_ds[0])
    print_line()

    # load model
    print("Loading and quantizing model...")
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16,
        attn_implementation=cfg.model.attn_implementation,
        # device_map="auto",
        # low_cpu_mem_usage=True,
    )
    if cfg.model.quantize_to_4bit:
        print("Quantizing model to 4-bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cfg.model.dtype
        )

    config_kwargs = dict(use_cache=False, pad_token_id=tokenizer.pad_token_id)
    if cfg.model.disable_attn_logit_softcapping:
        print("Setting attn_logit_softcapping to None")
        config_kwargs["attn_logit_softcapping"] = None
        # https://github.com/huggingface/transformers/blob/d5aebc64653d09660818109f2fac55b5e1031023/src/transformers/models/opt/modeling_opt.py#L263

    # if "sfairXC".lower() not in cfg.model.model_name.lower():
    # config_kwargs["num_labels"] = 2
    config = AutoConfig.from_pretrained(cfg.model.model_name, **config_kwargs)
    print(f"\nModel kwargs: {model_kwargs}")
    print(f"\nModel config: {config}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        config=config,
        **model_kwargs,
    )

    if cfg.model.reinit_head:
        print("Reinitializing the head")
        model.config.num_labels = 1
        model.score = nn.Linear(model.config.hidden_size, model.config.num_labels, bias=False)
        for name, param in model.score.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param.data)
            elif "head" in name:
                print(f"re-init {name}")
                nn.init.xavier_uniform_(param.data)
            param.requires_grad = True

    print_line()
    print("Model Architecture:")
    print(model)
    print_line()

    state.wait_for_everyone()

    print("Preparing model for PEFT")
    # attach lora adapters
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        target_modules=cfg_dict["lora"]["target_modules"],
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type=TaskType.SEQ_CLS,
        modules_to_save=cfg_dict["lora"]["modules_to_save"],
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    if main_process:
        model.print_trainable_parameters()

    # optimizer
    optimizer = get_optimizer(model, cfg.optimizer.name, cfg.optimizer.head_lr, cfg.optimizer.lr, cfg.trainer_args.weight_decay, print_fn=print)
    # training arguments
    training_args = RewardConfig(
        output_dir=out_dir,
        max_length=cfg.tokenizer.max_length,
        dataset_num_proc=num_proc,
        **cfg.trainer_args,
    )

    state.wait_for_everyone()

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor.tokenizer,
        optimizers=(optimizer, None),
    )

    trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    if main_process:
        # save omegaconf config
        OmegaConf.save(cfg, os.path.join(out_dir, "experiment_config.yaml"), resolve=True)
        # save training script
        shutil.copyfile(os.path.abspath(__file__), os.path.join(out_dir, "train.py"))

        # upload to hf hub
        if cfg.upload_to_hf_hub:
            repo_url = upload_artifacts_to_hf_hub(out_dir, "wsdm-cup", cfg.experiment_name, "./")
            print(f"Artifacts uploaded to {repo_url}")
            if cfg.use_wandb:
                wandb.run.summary["hf_hub_url"] = repo_url

        if notify_discord:
            discord.post(content=f"🎉 Experiment {cfg.experiment_name} completed in {asHours(time.time() - exp_start_time)}")


if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
