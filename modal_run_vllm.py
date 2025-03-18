import modal

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "bitsandbytes",
        "transformers",
        "datasets",
        "scikit-learn",
        "nvitop",
        "kagglehub",
        "peft",
        "huggingface_hub",
        "hf_transfer",
        "sentencepiece",
        "discordwebhook",
        "python-dotenv",
        "wandb",
        "omegaconf",
        "vllm",
    )
    .run_commands(  # add flash-attn
        "CXX=g++ pip install flash-attn --no-build-isolation"
    )
    .add_local_dir("/Users/atharva/Work/Kaggle/WSDM-Cup-Multilingual-Chatbot-Arena-Kaggle", remote_path="/root")
)
app = modal.App(image=image)
volume = modal.Volume.from_name("modal-examples-jupyter-inside-modal-data", create_if_missing=True)

CACHE_DIR = "/root/cache"


# This is all that's needed to create a long-lived Jupyter server process in Modal
# that you can access in your Browser through a secure network tunnel.
# This can be useful when you want to interactively engage with Volume contents
# without having to download it to your host computer.


@app.function(concurrency_limit=1, volumes={CACHE_DIR: volume}, timeout=84_400, gpu=modal.gpu.A10G(count=1))
def run_job(timeout: int):
    import subprocess
    import sys
    import torch

    print(f"GPU name: {torch.cuda.get_device_name()}")
    subprocess.run(["ls", "-l"], stdout=sys.stdout, stderr=sys.stderr, check=True)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ids_path", required=True, help="Path to conversation IDs pickle file")
    # parser.add_argument("--output_dir", required=True, help="Output parquet path")
    # parser.add_argument("--save_file_name", required=True, help="Output parquet file name")
    # parser.add_argument("--threads", type=int, default=8)
    # args = parser.parse_args()
    run_cmd = "python scripts/vllm_generate.py --ignore_ids_path ./data/vllm_inference_ignore_ids.pkl --save_name lmsys_1m_vllm_110k.parquet --out_dir ./artifacts"
    # run_cmd = "python scripts/merge_model.py"
    # run_cmd = "python scripts/pseudo_label.py"
    print(run_cmd)
    subprocess.run(
        run_cmd.split(),
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


@app.local_entrypoint()
def main(timeout: int = 84_400):
    run_job.remote(timeout=timeout)
