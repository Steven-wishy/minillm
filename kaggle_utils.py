import sys
import time


FINAL_KERNEL_STATUSES = frozenset(
    {"complete", "error", "cancel", "cancelled", "failure"}
)


def get_kernel_status(api, kernel_id: str) -> str:
    status = api.kernels_status(kernel_id).status
    try:
        return status.name.lower()
    except AttributeError:
        return str(status).lower()


def print_new_kernel_logs(api, kernel_id: str, printed_length: int) -> int:
    log_text = api.kernels_logs(kernel_id)
    if not log_text or len(log_text) <= printed_length:
        return printed_length

    print(log_text[printed_length:], end="")
    sys.stdout.flush()
    return len(log_text)


def stream_kernel_until_complete(
    api,
    kernel_id: str,
    poll_interval: int = 30,
    max_consecutive_errors: int = 25,
) -> str:
    printed_length = 0
    consecutive_errors = 0

    while True:
        try:
            status = get_kernel_status(api, kernel_id)
            consecutive_errors = 0
            print(f"\n[Status Update: {status}]")
            sys.stdout.flush()

            try:
                printed_length = print_new_kernel_logs(
                    api,
                    kernel_id,
                    printed_length,
                )
            except Exception:
                pass

            if status in FINAL_KERNEL_STATUSES:
                print(f"\n[Terminating tracking: final status is '{status}']")
                return status
        except Exception as api_error:
            consecutive_errors += 1
            print(
                f"\n[API Connection Warning (count={consecutive_errors})]: "
                f"{api_error}"
            )
            sys.stdout.flush()
            if consecutive_errors > max_consecutive_errors:
                raise RuntimeError(
                    "Too many consecutive Kaggle API connection errors."
                ) from api_error

        time.sleep(poll_interval)
