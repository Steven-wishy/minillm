import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_utils import stream_kernel_until_complete

def main():
    api = KaggleApi()
    api.authenticate()
    
    kernel_id = sys.argv[1] if len(sys.argv) > 1 else "shakuji/minillm-xsmall-finetuning"
    print(f"Resuming tracking for kernel '{kernel_id}'...")
    
    try:
        stream_kernel_until_complete(api, kernel_id)
    except RuntimeError as error:
        print(error)
        sys.exit(1)

if __name__ == '__main__':
    main()
