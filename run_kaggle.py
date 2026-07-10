import sys
import subprocess

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_utils import stream_kernel_until_complete

def main():
    api = KaggleApi()
    api.authenticate()
    
    kernel_id = "shakuji/minillm-xsmall-finetuning"
    kernel_dir = "/sdcard/Download/minillm/kaggle_run"
    print("Uploading and starting the T4 script execution on Kaggle...")
    
    # Push the kernel using CLI with explicit --accelerator flag
    # The metadata field alone doesn't reliably enforce T4 allocation
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", kernel_dir, "--accelerator", "NvidiaTeslaT4"],
        capture_output=True, text=True
    )
    print("Push stdout:", result.stdout.strip())
    if result.stderr:
        print("Push stderr:", result.stderr.strip())
    if result.returncode != 0:
        print(f"Push failed with exit code {result.returncode}")
        sys.exit(1)
    
    print("Tracking execution status...")
    try:
        stream_kernel_until_complete(api, kernel_id, poll_interval=20)
    except RuntimeError as error:
        print(error)
        sys.exit(1)

if __name__ == '__main__':
    main()
