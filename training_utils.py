from typing import Dict

import torch
import torch.nn.functional as F


def get_sequence_log_prob(model, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    """Compute the length-normalized completion log probability."""
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    full_tokens = tokenizer(prompt + completion, return_tensors="pt")

    prompt_len = prompt_tokens["input_ids"].shape[1]
    full_len = full_tokens["input_ids"].shape[1]

    if full_len <= prompt_len:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    input_ids = full_tokens["input_ids"].to(model.device)
    attention_mask = full_tokens["attention_mask"].to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    completion_logits = outputs.logits[0, prompt_len - 1 : full_len - 1, :]
    completion_targets = input_ids[0, prompt_len:full_len]

    log_probs = F.log_softmax(completion_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=completion_targets.unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs.mean()


def run_slerp_adapter_merge(
    model,
    weights_v0: Dict[str, torch.Tensor],
    weights_v1: Dict[str, torch.Tensor],
    t: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Merge two sets of LoRA adapter weights with SLERP."""
    merged_weights = {}
    for key in weights_v0:
        if key not in weights_v1:
            merged_weights[key] = weights_v0[key].clone()
            continue

        v0 = weights_v0[key]
        v1 = weights_v1[key]

        original_shape = v0.shape
        v0_flat = v0.view(-1).float()
        v1_flat = v1.view(-1).float()

        v0_norm = torch.norm(v0_flat)
        v1_norm = torch.norm(v1_flat)

        if v0_norm < 1e-8 or v1_norm < 1e-8:
            merged_flat = (1 - t) * v0_flat + t * v1_flat
        else:
            v0_unit = v0_flat / v0_norm
            v1_unit = v1_flat / v1_norm

            dot = torch.clamp(torch.dot(v0_unit, v1_unit), -1.0, 1.0)
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)

            if sin_omega < 1e-4:
                merged_flat = (1 - t) * v0_flat + t * v1_flat
            else:
                weight_v0 = torch.sin((1 - t) * omega) / sin_omega
                weight_v1 = torch.sin(t * omega) / sin_omega
                merged_flat = weight_v0 * v0_flat + weight_v1 * v1_flat

                target_norm = (1 - t) * v0_norm + t * v1_norm
                merged_flat = (merged_flat / torch.norm(merged_flat)) * target_norm

        merged_weights[key] = merged_flat.view(original_shape).to(v0.dtype)

    return merged_weights
