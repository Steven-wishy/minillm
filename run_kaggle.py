import time
import sys
import subprocess
import json
from kaggle.api.kaggle_api_extended import KaggleApi

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
    
    # Wait for execution and stream logs
    print("Tracking execution status...")
    last_printed_len = 0
    
    while True:
        status_result = api.kernels_status(kernel_id)
        if hasattr(status_result, 'status') and hasattr(status_result.status, 'name'):
            status = status_result.status.name.lower()
        elif hasattr(status_result, 'status'):
            status = str(status_result.status).lower()
        else:
            status = 'unknown'
        print(f"\n[Status Update: {status}]")
        
        try:
            log_text = api.kernels_logs(kernel_id)
            if log_text and len(log_text) > last_printed_len:
                new_text = log_text[last_printed_len:]
                print(new_text, end="")
                last_printed_len = len(log_text)
                sys.stdout.flush()
        except Exception as e:
            print(f"[Warning] Failed to fetch kernel logs this cycle: {e}", file=sys.stderr)
            sys.stderr.flush()
            
        if status in ['complete', 'error', 'cancel', 'cancelled', 'failure']:
            break
            
        time.sleep(20)

if __name__ == '__main__':
    main()
