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
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention

Gemma2Attention._flash_attn_uses_top_left_mask = False

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


def prepare_data(df, sample_data=False):
    if sample_data:
        df = df.sample(100, random_state=42)
    df["labels"] = df["winner"].map({"model_a": 0, "model_b": 1})
    # df["text"] = df.apply(prepare_text, axis=1)
    return df


def compute_metrics(eval_preds):
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    preds = torch.from_numpy(preds).float().softmax(-1).argmax(-1).numpy()
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {"accuracy": acc}


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


import torch.nn.functional as F
from transformers import LlamaModel, LlamaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class CustomLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config, model_name, quant_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.score = nn.Linear(config.hidden_size, 2, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)

        logits = logits[torch.arange(input_ids.shape[0], device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=transformer_outputs.hidden_states if output_hidden_states else None)


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


def main(cfg: DictConfig):
    exp_start_time = time.time()
    out_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    main_process = state.is_main_process

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
    df = prepare_data(df, cfg.sample_data)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "valid"].copy().reset_index(drop=True)

    start_msg = f"🚀 Starting experiment {cfg.experiment_name}"
    if notify_discord and main_process:
        discord.post(content=start_msg)

    print_line()
    print(start_msg)
    print("Configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))
    print_line()

    # tokenize data
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(val_df)
    processor = SequenceProcessor(
        cfg.model.model_name,
        max_length=cfg.tokenizer.max_length,
        truncation_side=cfg.tokenizer.truncation_side,
        padding_side=cfg.tokenizer.padding_side,
    )
    tokenizer = processor.tokenizer
    with state.main_process_first():
        train_ds = train_ds.map(
            processor.tokenize,
            batched=False,
            num_proc=8,
            fn_kwargs={"tta": False},
            desc="Tokenizing train data",
        )
        eval_ds = eval_ds.map(
            processor.tokenize,
            batched=False,
            num_proc=8,
            fn_kwargs={"tta": False},
            desc="Tokenizing eval data",
        )

    # check how many of the samples exceed the max_length
    print_line()
    print("Checking how many samples exceed the max_length")
    train_exceed = train_ds.filter(lambda x: x["length"] > cfg.tokenizer.max_length)
    eval_exceed = eval_ds.filter(lambda x: x["length"] > cfg.tokenizer.max_length)
    print(f"Train samples exceeding max_length: {len(train_exceed)}")
    print(f"Eval samples exceeding max_length: {len(eval_exceed)}")
    print_line()

    # print sample data
    print_line()
    print("Sample Data:")
    print(f"Train Data: {len(train_ds)} samples | Eval Data: {len(eval_ds)} samples")
    print_line()
    print(train_ds[0])
    print("\nFormatted Text:\n")
    print(tokenizer.decode(train_ds[0]["input_ids"]))
    # print random sample
    print_line()
    rand_idx = np.random.randint(0, len(train_ds))
    print(train_ds[rand_idx])
    print("\nFormatted Text:\n")
    print(tokenizer.decode(train_ds[rand_idx]["input_ids"]))
    print_line()

    # load model
    print("Loading and quantizing model...")
    if cfg.model.quantize_to_4bit:
        print("Quantizing model to 4-bit")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cfg.model.dtype
        )

    config = AutoConfig.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    model = CustomLlamaModel(config=config, model_name=cfg.model.model_name, quant_config=quant_config)
    model.model.config.pad_token_id = tokenizer.pad_token_id

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
    training_args = TrainingArguments(
        output_dir=out_dir,
        **cfg.trainer_args,
    )

    state.wait_for_everyone()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        optimizers=(optimizer, None),
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=cfg.tokenizer.pad_to_multiple_of),
        compute_metrics=MetricCalculator(val_df, out_dir),
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
