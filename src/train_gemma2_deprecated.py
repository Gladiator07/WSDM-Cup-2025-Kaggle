import os
import sys
import time
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # for ultra-fast downloads

import huggingface_hub as hf_hub
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from accelerate.state import PartialState
from datasets import Dataset
from dotenv import load_dotenv
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from simple_parsing.helpers import Serializable
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
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


# ------------------------- Config -------------------------
@dataclass
class Config(Serializable):
    # paths
    output_dir: str = "./artifacts"
    data_dir: str = "./data"

    # misc
    use_wandb: bool = True
    upload_to_hf_hub: bool = True
    debug: bool = False
    seed: int = -1

    experiment_name: str = "gemma2_rm"
    notes: str = "use flash attention 2, increase max_length to 1536 and start training with reward model"

    # data
    train_file_name: str = "train_split.parquet"

    # model
    model_name: str = "sfairXC/FsfairX-Gemma2-RM-v0.1"
    load_in_4bit: bool = False

    # tokenizer
    max_length: int = 1536
    truncation: bool = True
    padding_side: str = "left"
    truncation_side: str = "left"

    # lora
    lora_r: int = 64
    lora_alpha: int = 64

    # optimizer
    lr: float = 7e-5
    head_lr: float = 1e-5

    # trainer
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    gradient_checkpointing: bool = True
    torch_compile: bool = True


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


# ------------------------- Model -------------------------
from transformers.models.gemma2 import Gemma2PreTrainedModel, Gemma2Model
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.gemma2 import Gemma2Config

# monkey patching Gemma2Attention to use bottom right mask to enable flash attention 2
Gemma2Attention._flash_attn_uses_top_left_mask = False


class CustomGemma2ForSequenceClassification(Gemma2PreTrainedModel):
    def __init__(self, config: Gemma2Config, load_in_4bit=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
        self.score = nn.Linear(config.hidden_size, 2, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
        )


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


def main(cfg: Config):
    exp_start_time = time.time()
    out_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    cfg_dict = cfg.to_dict()
    main_process = state.is_main_process

    if notify_discord and main_process:
        discord = Discord(url=os.getenv("DISCORD_WEBHOOK"))

    if cfg.seed == -1:
        cfg.seed = np.random.randint(0, 1000)
    set_seed(cfg.seed)

    if cfg.debug:
        cfg.upload_to_hf_hub = False
        sample_data = True

    if main_process:
        wandb.init(name=cfg.experiment_name, notes=cfg.notes, config=cfg_dict)

    # load data
    df = pd.read_parquet(os.path.join(cfg.data_dir, cfg.train_file_name))
    df = prepare_data(df, sample_data)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "valid"].copy().reset_index(drop=True)

    start_msg = f"🚀 Starting experiment {cfg.experiment_name}"
    if notify_discord and main_process:
        discord.post(content=start_msg)

    print_line()
    print(start_msg)
    print("Configuration:\n", cfg_dict)
    print_line()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=True,
        from_slow=True,
        add_prefix_space=False,
        padding_side=cfg.padding_side,
        truncation_side=cfg.truncation_side,
    )
    tokenizer.add_eos_token = False

    # tokenize data
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(val_df)
    with state.main_process_first():
        train_ds = train_ds.map(
            tokenize_func,
            batched=True,
            num_proc=8,
            fn_kwargs={"tokenizer": tokenizer, "truncation": cfg.truncation, "max_length": cfg.max_length},
            desc="Tokenizing train data",
        )
        eval_ds = eval_ds.map(
            tokenize_func,
            batched=True,
            num_proc=8,
            fn_kwargs={"tokenizer": tokenizer, "truncation": cfg.truncation, "max_length": cfg.max_length},
            desc="Tokenizing eval data",
        )

    # print sample data
    print_line()
    print("Sample Data:")
    print(f"Train Data: {len(train_ds)} samples | Eval Data: {len(eval_ds)} samples")
    print(train_ds[0])
    print_line()

    # load model
    print("Loading model...")
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # device_map="auto",
        # low_cpu_mem_usage=True,
    )
    if cfg.model.quantize_to_4bit:
        print("Quantizing model to 4-bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cfg.model.dtype
        )

    config_kwargs = dict(use_cache=False, pad_token_id=tokenizer.pad_token_id)
    if "gemma" in cfg.model.model_name.lower():
        print("Setting attn_logit_softcapping to None for GEMMA model")
        config_kwargs["attn_logit_softcapping"] = None
        # https://github.com/huggingface/transformers/blob/d5aebc64653d09660818109f2fac55b5e1031023/src/transformers/models/opt/modeling_opt.py#L263

    if "sfairXC".lower() not in cfg.model.model_name.lower():
        config_kwargs["num_labels"] = 2

    config = AutoConfig.from_pretrained(cfg.model.model_name, **config_kwargs)
    config._flash_attn_uses_top_left_mask = False
    print(f"\nModel kwargs: {model_kwargs}")
    print(f"\nModel config: {config}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        config=config,
        **model_kwargs,
    )

    if "sfairXC".lower() in cfg.model.model_name.lower():
        print("sfairXC model detected. Changing the head to 2 classes")
        model.score = nn.Linear(model.config.hidden_size, 2, bias=False)

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

        # upload to hf hub
        if cfg.upload_to_hf_hub:
            repo_url = upload_artifacts_to_hf_hub(out_dir, "wsdm-cup", cfg.experiment_name, "./")
            print(f"Artifacts uploaded to {repo_url}")
            if cfg.use_wandb:
                wandb.run.summary["hf_hub_url"] = repo_url

        if notify_discord:
            discord.post(content=f"🎉 Experiment {cfg.experiment_name} completed in {asHours(time.time() - exp_start_time)}")


if __name__ == "__main__":
    import simple_parsing

    cfg: Config = simple_parsing.parse(Config)

    os.environ["WANDB_PROJECT"] = "WSDM-Cup-Multilingual-Chatbot-Arena"
    # disable wandb if false
    if not cfg.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    main(cfg)
