import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # for ultra-fast downloads
import huggingface_hub as hf_hub
import torch
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

MODEL_ID = "sfairXC/FsfairX-Gemma2-RM-v0.1"
LORA_REPO_ID = "Gladiator/wsdm_cup_gemma2_9b_stage2_v9"
DOWNLOAD_DIR = f"./artifacts/{LORA_REPO_ID.split('/')[1]}"
MERGED_SAVE_DIR = "./artifacts/merged_model"
HUB_NAME = "gemma2_9b_stage2_v9_merged"


def upload_artifacts_to_hf_hub(upload_dir: str, repo_id_prefix: str, experiment_name: str, path_in_repo: str):
    hf_hub.login(token=os.environ["HF_TOKEN"], write_permission=True)
    api = hf_hub.HfApi()
    repo_id = f"{os.environ['HF_USERNAME']}/{repo_id_prefix}_{experiment_name}"
    repo_url = hf_hub.create_repo(repo_id, exist_ok=True, private=True)
    api.upload_folder(folder_path=upload_dir, repo_id=repo_id, path_in_repo=path_in_repo)
    return repo_url


snapshot_download(repo_id=LORA_REPO_ID, local_dir=DOWNLOAD_DIR)
adapter_path = os.path.join(DOWNLOAD_DIR)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.score = nn.Linear(model.config.hidden_size, 2, bias=False, dtype=torch.bfloat16)
model.config.num_labels = 2
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

peft_model = PeftModel.from_pretrained(model, adapter_path)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(MERGED_SAVE_DIR, safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(MERGED_SAVE_DIR)

os.makedirs(os.path.join(MERGED_SAVE_DIR, "exp_metadata"), exist_ok=True)
os.system(f"cp -r {DOWNLOAD_DIR}/* {MERGED_SAVE_DIR}/exp_metadata/")

upload_artifacts_to_hf_hub(upload_dir=MERGED_SAVE_DIR, repo_id_prefix="wsdm_cup", experiment_name=HUB_NAME, path_in_repo="./")
