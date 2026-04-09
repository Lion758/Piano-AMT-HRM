import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def is_no_decay_parameter(name):
    lowered = name.lower()
    if lowered.endswith("bias"):
        return True

    no_decay_tokens = (
        "norm",
        "layer_norm",
        "layernorm",
        "rms_norm",
        "rmsnorm",
        "embedding",
        "embed",
    )
    return any(token in lowered for token in no_decay_tokens)


def build_trm_optimizer(model, learning_rate, weight_decay):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_decay_parameter(name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return AdamW(param_groups, learning_rate)


def build_trm_scheduler(optimizer, training_steps, warmup_ratio, min_lr_ratio):
    total_steps = max(1, int(training_steps))
    warmup_steps = int(round(total_steps * warmup_ratio))
    warmup_steps = min(max(warmup_steps, 0), total_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(warmup_steps)

        if total_steps == warmup_steps:
            return min_lr_ratio

        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
