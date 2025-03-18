# ---
# args: ["--timeout", 10]
# ---

# ## Overview
#
# Quick snippet showing how to connect to a Jupyter notebook server running inside a Modal container,
# especially useful for exploring the contents of Modal Volumes.
# This uses [Modal Tunnels](https://modal.com/docs/guide/tunnels#tunnels-beta)
# to create a tunnel between the running Jupyter instance and the internet.
#
# If you want to your Jupyter notebook to run _locally_ and execute remote Modal Functions in certain cells, see the `basic.ipynb` example :)


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
        "trl",
        "liger-kernel",
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


@app.function(concurrency_limit=1, volumes={CACHE_DIR: volume}, timeout=84_400, gpu=modal.gpu.H100(count=2))
def run_training(timeout: int):
    import subprocess
    import sys
    import torch

    print(f"total gpus: {torch.cuda.device_count()}")
    print(f"gpu name 0: {torch.cuda.get_device_name(0)}")
    print(f"gpu name 1: {torch.cuda.get_device_name(1)}")
    subprocess.run(["pip", "install", "hf_transfer"], stdout=sys.stdout, stderr=sys.stderr, check=True)
    # install bitsandbytes
    subprocess.run(["pip", "install", "-U", "bitsandbytes"], stdout=sys.stdout, stderr=sys.stderr, check=True)
    # subprocess.run(["pip", "install", "liger-kernel"], stdout=sys.stdout, stderr=sys.stderr, check=True)
    # list current files and directories
    subprocess.run(["ls", "-l"], stdout=sys.stdout, stderr=sys.stderr, check=True)

    run_cmd = "accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=2 --dynamo_backend=inductor src/train_gemma2.py configs/gemma2_9B_stage1_v3.yaml"
    subprocess.run(
        run_cmd.split(),
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


@app.local_entrypoint()
def main(timeout: int = 84_400):
    run_training.remote(timeout=timeout)
