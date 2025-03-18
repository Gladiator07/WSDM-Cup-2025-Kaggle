import os
import sys
import time
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # for ultra-fast downloads
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if True:  # for ignoring ruff warnings
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
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoConfig,
        BitsAndBytesConfig,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

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
def prepare_text(row):
    prompt, res_a, res_b = row["prompt"], row["response_a"], row["response_b"]
    text = "## Prompt\n" + prompt + "\n\n## Response A\n" + res_a + "\n\n## Response B\n" + res_b + "\n\n" + "## Which response is better?"
    return text


def prepare_data(df, sample_data=False):
    if sample_data:
        df = df.sample(100, random_state=42)
    df["labels"] = df["winner"].map({"model_a": 0, "model_b": 1})
    df["text"] = df.apply(prepare_text, axis=1)
    return df


def tokenize_func(example, tokenizer, truncation, max_length):
    return tokenizer(example["text"], truncation=truncation, max_length=max_length)


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


def get_optimizer(model, optimizer_name, head_lr, lr, weight_decay, print_fn=print):
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

    if notify_discord:
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

    if cfg.use_wandb:
        # delete trainer_args from cfg_to_log
        cfg_dict.pop("trainer_args")
        wandb.init(project=cfg.wandb_project_name, name=cfg.experiment_name, notes=cfg.notes, config=cfg_dict)

    # load data
    df = pd.read_parquet(os.path.join(cfg.data_dir, cfg.train_file_name))
    df = prepare_data(df, cfg.sample_data)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "valid"].copy().reset_index(drop=True)

    start_msg = f"🚀 Starting experiment {cfg.experiment_name}"
    discord.post(content=start_msg)

    print_line()
    print(start_msg)
    print("Configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))
    print_line()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        use_fast=True,
        trust_remote_code=True,
        from_slow=True,
        add_prefix_space=False,
        padding_side=cfg.tokenizer.padding_side,
        truncation_side=cfg.tokenizer.truncation_side,
    )
    tokenizer.add_eos_token = True

    # tokenize data
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(val_df)
    train_ds = train_ds.map(
        tokenize_func,
        batched=True,
        num_proc=8,
        fn_kwargs={"tokenizer": tokenizer, "truncation": cfg.tokenizer.truncation, "max_length": cfg.tokenizer.max_length},
        desc="Tokenizing train data",
    )
    eval_ds = eval_ds.map(
        tokenize_func,
        batched=True,
        num_proc=8,
        fn_kwargs={"tokenizer": tokenizer, "truncation": cfg.tokenizer.truncation, "max_length": cfg.tokenizer.max_length},
        desc="Tokenizing eval data",
    )

    # print sample data
    print_line()
    print("Sample Data:")
    print(f"Train Data: {len(train_ds)} samples | Eval Data: {len(eval_ds)} samples")
    print(train_ds[0])
    print_line()

    # load model
    print("Loading and quantizing model...")
    config = AutoConfig.from_pretrained(cfg.model.model_name, trust_remote_code=True, num_labels=2, use_cache=False)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )
    if cfg.model.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cfg.model.dtype
        )
    print(f"Model kwargs: {model_kwargs}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        config=config,
        **model_kwargs,
    )
    if "gemma" in cfg.model.model_name.lower():
        print("Using GEMMA model, setting attn_logit_softcapping to None")
        model.config.attn_logit_softcapping = None

    model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.use_cache =   False

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

    model.print_trainable_parameters()

    # optimizer
    optimizer = get_optimizer(model, cfg.optimizer.name, cfg.optimizer.head_lr, cfg.optimizer.lr, cfg.trainer_args.weight_decay, print_fn=print)
    # training arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        **cfg.trainer_args,
    )

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

    # save omegaconf config
    OmegaConf.save(cfg, os.path.join(out_dir, "experiment_config.yaml"), resolve=True)

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
