import argparse
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import huggingface_hub as hf_hub
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from tqdm import tqdm

load_dotenv()

MODELS = [
    # "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "meta-llama/Llama-3.1-70B-Instruct",
    "01-ai/Yi-1.5-34B-Chat",
    # "Qwen/QwQ-32B-Preview",
    "HuggingFaceH4/starchat2-15b-v0.1",
    # "google/gemma-2-27b-it",
    # "meta-llama/Meta-Llama-3-8B-Instruct",
]


def upload_artifacts_to_hf_hub(upload_dir: str, repo_id_prefix: str, experiment_name: str, path_in_repo: str):
    hf_hub.login(token=os.environ["HF_TOKEN"], write_permission=True)
    api = hf_hub.HfApi()
    repo_id = f"{os.environ['HF_USERNAME']}/{repo_id_prefix}_{experiment_name}"
    repo_url = hf_hub.create_repo(repo_id, exist_ok=True, private=True)
    api.upload_folder(folder_path=upload_dir, repo_id=repo_id, path_in_repo=path_in_repo)
    return repo_url


def load_and_filter_dataset(ids_path):
    # Load conversation IDs
    with open(ids_path, "rb") as f:
        conv_ids = pickle.load(f)

    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
    dataset = dataset.filter(lambda x: x["conversation_id"] in conv_ids)
    print(f"Filtered dataset to {len(dataset)} conversations")
    data = []

    for item in dataset:
        prompt = next(msg["content"] for msg in item["conversation"] if msg["role"] == "user")
        orig_response = next(msg["content"] for msg in item["conversation"] if msg["role"] == "assistant")

        # Randomly assign responses
        if random.random() < 0.5:
            data.append(
                {
                    "id": item["conversation_id"],
                    "prompt": prompt,
                    "response_a": orig_response,
                    "response_b": "",
                    "model_a": item["model"],
                    "model_b": "",
                    "language": item["language"],
                }
            )
        else:
            data.append(
                {
                    "id": item["conversation_id"],
                    "prompt": prompt,
                    "response_a": "",
                    "response_b": orig_response,
                    "model_a": "",
                    "model_b": item["model"],
                    "language": item["language"],
                }
            )

    return pd.DataFrame(data)


def process_row(row, client):
    try:
        model = random.choice(MODELS)
        messages = [{"role": "user", "content": row["prompt"]}]
        max_tokens = 2560 if model == "01-ai/Yi-1.5-34B-Chat" else 8000
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=0.7, top_p=0.9)

        response_content = response.choices[0].message.content
        time.sleep(1)

        return response_content, model

    except Exception as e:
        print(f"Error processing row {row['id']}: {str(e)}")
        return "", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_path", required=True, help="Path to conversation IDs pickle file")
    parser.add_argument("--output_dir", required=True, help="Output parquet path")
    parser.add_argument("--save_file_name", required=True, help="Output parquet file name")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    print("Loading and preparing dataset...")
    df = load_and_filter_dataset(args.ids_path)
    total_rows = len(df)
    print(f"Loaded {total_rows} conversations")

    client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
    processed_count = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Only process rows that need responses
        futures = {executor.submit(process_row, row, client): idx for idx, row in df.iterrows() if (not row["response_a"] or not row["response_b"])}

        with tqdm(total=len(futures), desc="Processing rows") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    response_content, model = future.result()
                    # Update the empty response column
                    if not df.at[idx, "response_a"]:
                        df.at[idx, "response_a"] = response_content
                        df.at[idx, "model_a"] = model
                    else:
                        df.at[idx, "response_b"] = response_content
                        df.at[idx, "model_b"] = model
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing result for index {idx}: {str(e)}")

                pbar.update(1)
                print(f"Progress: {processed_count}/{len(futures)} rows completed")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.save_file_name)
    df.to_parquet(save_path, index=False)
    print(f"Saved results to {save_path}")

    print("Uploading results to HF Hub...")
    upload_artifacts_to_hf_hub(args.output_dir, "wsdm_cup", args.save_file_name.split(".")[0], args.save_file_name)


if __name__ == "__main__":
    main()
