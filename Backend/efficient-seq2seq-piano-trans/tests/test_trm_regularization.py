import math
import sys
import types
import warnings
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

data_module = types.ModuleType("data")
data_module.__path__ = []
constants_module = types.ModuleType("data.constants")
constants_module.TOKEN_PAD = 0
sys.modules.setdefault("data", data_module)
sys.modules["data.constants"] = constants_module

from loss.CrossEntropyLoss import CrossEntropyLoss
from utils.trm_training import build_trm_optimizer, build_trm_scheduler


class _DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.layer_norm = nn.LayerNorm(4)
        self.token_embed = nn.Embedding(8, 4)


def _make_config(encoder_name):
    model = type(
        "ModelConfig",
        (),
        {
            "model_name": "Transformer-T5",
            "encoder_name": encoder_name,
            "vocab_size": 4,
            "checkpoint_ignore_layres": [],
        },
    )()
    training = type(
        "TrainingConfig",
        (),
        {
            "learning_rate": 1e-4,
            "training_steps": 160000,
            "trm_weight_decay": 0.01,
            "trm_warmup_ratio": 0.05,
            "trm_cosine_min_lr_ratio": 0.1,
            "trm_label_smoothing": 0.05,
            "get": lambda self, key, default=None: getattr(self, key, default),
        },
    )()
    return type("Config", (), {"model": model, "training": training})()


def test_cross_entropy_loss_disables_label_smoothing_for_non_trm():
    config = _make_config("TransformerEncoder")

    criterion = CrossEntropyLoss(config)

    assert criterion.criterion.label_smoothing == 0.0


def test_cross_entropy_loss_enables_label_smoothing_for_trm_and_ignores_pad():
    config = _make_config("TrmEncoder")

    criterion = CrossEntropyLoss(config)
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0, -2.0], [0.5, 1.0, -0.5, -1.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[1, 2]], dtype=torch.long)
    targets_mask = torch.tensor([[1, 0]], dtype=torch.long)

    loss = criterion(logits, targets, targets_mask)

    valid_logits = logits[:, :1, :]
    valid_targets = targets[:, :1]
    expected = nn.CrossEntropyLoss(label_smoothing=0.05)(valid_logits.transpose(1, 2), valid_targets)

    assert criterion.criterion.label_smoothing == 0.05
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected)

def test_non_trm_optimizer_path_is_unchanged():
    model = _DummyTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_trm_optimizer_uses_split_weight_decay_and_step_scheduler():
    config = _make_config("TrmEncoder")
    model = _DummyTransformer()
    optimizer = build_trm_optimizer(model, config.training.learning_rate, config.training.trm_weight_decay)
    scheduler = build_trm_scheduler(
        optimizer,
        config.training.training_steps,
        config.training.trm_warmup_ratio,
        config.training.trm_cosine_min_lr_ratio,
    )

    weight_decays = {group["weight_decay"] for group in optimizer.param_groups}
    warmup_steps = round(config.training.training_steps * config.training.trm_warmup_ratio)

    assert weight_decays == {0.0, 0.01}
    assert warmup_steps == 8000
    assert math.isclose(optimizer.defaults["lr"] * config.training.trm_cosine_min_lr_ratio, 1e-5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler.step(warmup_steps)
    assert math.isclose(optimizer.param_groups[0]["lr"], optimizer.defaults["lr"], rel_tol=1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler.step(config.training.training_steps)
    assert math.isclose(
        optimizer.param_groups[0]["lr"],
        optimizer.defaults["lr"] * config.training.trm_cosine_min_lr_ratio,
        rel_tol=1e-6,
    )


def test_trm_scheduler_state_restores_progress():
    config = _make_config("TrmEncoder")
    optimizer_a = build_trm_optimizer(_DummyTransformer(), config.training.learning_rate, config.training.trm_weight_decay)
    scheduler_a = build_trm_scheduler(
        optimizer_a,
        config.training.training_steps,
        config.training.trm_warmup_ratio,
        config.training.trm_cosine_min_lr_ratio,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler_a.step(12345)
    saved_optimizer_state = optimizer_a.state_dict()
    saved_scheduler_state = scheduler_a.state_dict()
    saved_lr = optimizer_a.param_groups[0]["lr"]

    optimizer_b = build_trm_optimizer(_DummyTransformer(), config.training.learning_rate, config.training.trm_weight_decay)
    scheduler_b = build_trm_scheduler(
        optimizer_b,
        config.training.training_steps,
        config.training.trm_warmup_ratio,
        config.training.trm_cosine_min_lr_ratio,
    )

    optimizer_b.load_state_dict(saved_optimizer_state)
    scheduler_b.load_state_dict(saved_scheduler_state)

    assert scheduler_b.last_epoch == scheduler_a.last_epoch
    assert math.isclose(optimizer_b.param_groups[0]["lr"], saved_lr, rel_tol=1e-6)
