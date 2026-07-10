import time
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    
    kernel_id = sys.argv[1] if len(sys.argv) > 1 else "shakuji/minillm-xsmall-finetuning"
    print(f"Resuming tracking for kernel '{kernel_id}'...")
    
    last_printed_len = 0
    consecutive_errors = 0
    
    while True:
        try:
            status_result = api.kernels_status(kernel_id)
            if hasattr(status_result, 'status') and hasattr(status_result.status, 'name'):
                status = status_result.status.name.lower()
            elif hasattr(status_result, 'status'):
                status = str(status_result.status).lower()
            else:
                status = 'unknown'
            
            # Reset error count on successful API call
            consecutive_errors = 0
            
            print(f"\n[Status Update: {status}]")
            sys.stdout.flush()
            
            try:
                log_text = api.kernels_logs(kernel_id)
                if log_text and len(log_text) > last_printed_len:
                    new_text = log_text[last_printed_len:]
                    print(new_text, end="")
                    last_printed_len = len(log_text)
                    sys.stdout.flush()
            except Exception as e:
                # Log reading error is non-fatal, but surface it so silent
                # log-streaming gaps are diagnosable.
                print(f"[Warning] Failed to fetch kernel logs this cycle: {e}", file=sys.stderr)
                sys.stderr.flush()
                
            if status in ['complete', 'error', 'cancel', 'cancelled', 'failure']:
                print(f"\n[Terminating tracking: final status is '{status}']")
                break
                
        except Exception as api_err:
            consecutive_errors += 1
            print(f"\n[API Connection Warning (count={consecutive_errors})]: {api_err}")
            sys.stdout.flush()
            if consecutive_errors > 25:
                print("Too many consecutive connection errors. Terminating tracker.")
                sys.exit(1)
                
        time.sleep(30)

if __name__ == '__main__':
    main()
