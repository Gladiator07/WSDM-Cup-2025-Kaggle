import argparse
import gc
import os
import pickle
import random
from collections import Counter

import pandas as pd
import torch
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from vllm import LLM, SamplingParams

load_dotenv()
random.seed(0)

hf_api = HfApi()

# models = [
#     # "Qwen/Qwen2.5-32B-Instruct-AWQ",  # pass
#     "google/gemma-2-9b-it",  # pass
#     "Qwen/Qwen2.5-14B-Instruct-AWQ",  # pass
#     # "01-ai/Yi-1.5-9B-Chat",  # pass
#     "meta-llama/Meta-Llama-3-8B-Instruct",  # pass
#     "meta-llama/Llama-3.2-3B-Instruct",  # pass
#     "meta-llama/Llama-3.1-8B-Instruct",  # pass
#     "Qwen/Qwen2.5-7B-Instruct",  # pass
#     "Nexusflow/Starling-LM-7B-beta",  # pass
#     "Qwen/Qwen2-7B-Instruct",  # pass
#     # "microsoft/Phi-3-small-8k-instruct", # fail (unpack error)
#     # "meta-llama/Llama-3.2-1B", # fail (chat template issue)
#     "Qwen/Qwen2-1.5B-Instruct",  # pass
#     # "alpaca-13b", # fail
#     "mistralai/Mistral-7B-Instruct-v0.2",  # pass
#     "mistralai/Mistral-7B-Instruct-v0.3",  # pass
#     "microsoft/Phi-3-mini-4k-instruct",  # pass
#     "allenai/Llama-3.1-Tulu-3-8B",  # pass
#     "google/gemma-2-2b-it",  # pass
#     "Qwen/Qwen2.5-3B-Instruct",  # pass
# ]
models = ["allenai/Llama-3.1-Tulu-3-8B"]

print(os.environ["HF_TOKEN"])


def select_model(example):
    return {"gen_model": random.choice(models)}


def get_prompt(example):
    return {"prompt": next(msg["content"] for msg in example["conversation"] if msg["role"] == "user")}


def format_to_dataframe(ds):
    df = []
    for f in ds:
        # Randomly decide which response goes to A and B
        orig_response = next(msg["content"] for msg in f["conversation"] if msg["role"] == "assistant")
        if random.random() < 0.5:
            response_a = f["generated_response"]
            response_b = orig_response
            model_a = f["gen_model"]
            model_b = f["model"]
        else:
            response_a = orig_response
            response_b = f["generated_response"]
            model_a = f["model"]
            model_b = f["gen_model"]

        df.append(
            {
                "id": f["conversation_id"],
                "prompt": f["prompt"],
                "response_a": response_a,
                "response_b": response_b,
                "model_a": model_a,
                "model_b": model_b,
                "language": f["language"],
            }
        )
    return pd.DataFrame(df)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ignore_ids_path, "rb") as f:
        ignore_ids = pickle.load(f)

    print(f"Ignoring {len(ignore_ids)} conversation IDs")

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
    if args.debug:
        ds = ds.select(range(5000))

    ds = ds.filter(lambda example: example["language"] != "English", num_proc=8)
    ds = ds.filter(lambda example: example["turn"] == 1, num_proc=8)
    ds = ds.filter(lambda x: x["conversation_id"] not in ignore_ids, batch_size=1000, num_proc=8)
    ds = ds.map(select_model, num_proc=8)
    ds = ds.map(get_prompt, num_proc=8)

    all_models_instances = [d["gen_model"] for d in ds]
    print("Model distribution:", Counter(all_models_instances))

    all_ds = []
    for model in models:
        try:
            max_model_len = 8192
            max_num_seqs = 256
            if model in ["01-ai/Yi-1.5-9B-Chat", "microsoft/Phi-3-mini-4k-instruct"]:
                max_model_len = 4096
            elif model == "HuggingFaceTB/SmolLM-1.7B-Instruct":
                max_model_len = 2048
            elif "9B" in model or "14B" in model:
                max_model_len = 4096
                max_num_seqs = 128
            print(f"model: {model}")
            llm = LLM(
                model=model,
                dtype="auto",
                trust_remote_code=True,
                quantization="awq" if "awq" in model.lower() else None,
                seed=0,
                gpu_memory_utilization=0.95,
                max_model_len=max_model_len,
                enable_chunked_prefill=True,
                max_num_seqs=max_num_seqs,
            )
            sampling_params = SamplingParams(n=1, temperature=0.7, max_tokens=4096, seed=0, top_p=0.9)

            # filter on gen_model
            model_ds = ds.filter(lambda x: x["gen_model"] == model, num_proc=8)
            print(f"Processing {len(model_ds)} samples for {model}")

            conversations = []
            for d in model_ds:
                conversations.append(
                    [
                        {"role": "user", "content": d["prompt"]},
                    ]
                )

            outputs = llm.chat(conversations, sampling_params=sampling_params)
            responses = []
            for o in outputs:
                response = o.outputs[0].text.strip()
                responses.append(response)
            model_ds = model_ds.add_column(name="generated_response", column=responses)

            model_df = format_to_dataframe(model_ds)
            print("Model dataframe shape:", model_df.shape)
            mname = model.replace("/", "_")
            save_name = f"{mname}_vllm_gen.parquet"
            save_path = os.path.join(args.out_dir, save_name)
            model_df.to_parquet(save_path, index=False)

            hf_api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=save_name,
                repo_id="Gladiator/wsdm_cup_lmsys_1m_vllm",
                repo_type="model",
            )

            all_ds.append(model_ds)

            del llm
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Failed for model {model}")
            print(str(e))
            try:
                # try deleting the model instance
                del llm
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(str(e))

    final_ds = concatenate_datasets(all_ds)
    final_df = format_to_dataframe(final_ds)
    print("Final dataframe shape:", final_df.shape)
    print("Final dataframe head:")
    print(final_df.head())
    save_path = os.path.join(args.out_dir, args.save_name)
    final_df.to_parquet(save_path)
    hf_api.upload_file(
        path_or_fileobj=save_path,
        path_in_repo=args.save_name,
        repo_id="Gladiator/wsdm_cup_lmsys_1m_vllm",
        repo_type="model",
    )
    print("Uploaded to hugginface hub")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLLM inference with multiple models")
    parser.add_argument("--ignore_ids_path", type=str, default="vllm_inference_ids.pkl", help="Path to pickle file containing IDs to ignore")
    parser.add_argument("--save_name", type=str, default="lmsys_1m_120k_vllm_gen.parquet", help="Name of the output file")
    parser.add_argument("--out_dir", type=str, default="./artifacts", help="Output directory for artifacts")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args)
