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

image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "pandas", "numpy", "huggingface_hub", "python-dotenv", "pyarrow", "datasets")
    .add_local_dir("/Users/atharva/Work/Kaggle/WSDM-Cup-Multilingual-Chatbot-Arena-Kaggle", remote_path="/root")
)
app = modal.App(image=image)
volume = modal.Volume.from_name("modal-examples-jupyter-inside-modal-data", create_if_missing=True)

CACHE_DIR = "/root/cache"


# This is all that's needed to create a long-lived Jupyter server process in Modal
# that you can access in your Browser through a secure network tunnel.
# This can be useful when you want to interactively engage with Volume contents
# without having to download it to your host computer.


@app.function(concurrency_limit=1, volumes={CACHE_DIR: volume}, timeout=84_400)
def run_job(timeout: int):
    import subprocess
    import sys

    subprocess.run(["ls", "-l"], stdout=sys.stdout, stderr=sys.stderr, check=True)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ids_path", required=True, help="Path to conversation IDs pickle file")
    # parser.add_argument("--output_dir", required=True, help="Output parquet path")
    # parser.add_argument("--save_file_name", required=True, help="Output parquet file name")
    # parser.add_argument("--threads", type=int, default=8)
    # args = parser.parse_args()
    run_cmd = "python scripts/hf_inference_generate.py --ids_path ./data/hf_serverless_inference_ids.pkl --output_dir ./output --save_file_name lmsys_1m_hf_serverless_15k.parquet --threads 8"
    subprocess.run(
        run_cmd.split(),
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


@app.local_entrypoint()
def main(timeout: int = 84_400):
    run_job.remote(timeout=timeout)
