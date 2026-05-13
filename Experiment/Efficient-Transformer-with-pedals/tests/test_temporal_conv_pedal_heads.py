from pathlib import Path
from contextlib import contextmanager
import importlib.util
import sys
import types

import torch
import torch.nn as nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
_MISSING = object()


def _module(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


@contextmanager
def _preserve_modules(module_names):
    saved_modules = {name: sys.modules.get(name, _MISSING) for name in module_names}
    try:
        yield
    finally:
        for name, module in saved_modules.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _install_t5_import_stubs():
    encoder_module = _module("model.Encoder")
    encoder_module.Encoder = type("Encoder", (nn.Module,), {})

    decoder_module = _module("model.Decoder")
    decoder_module.Decoder = type("Decoder", (nn.Module,), {})
    decoder_module.CompoundDecoder = type("CompoundDecoder", (nn.Module,), {})

    layers_module = _module("model.Layers")
    layers_module.normalize_transformer_ffn_activation = lambda activation: activation

    _module("model.Mask")

    hppnet_module = _module("model.HPPNet")
    hppnet_module.HPPNet = type("HPPNet", (nn.Module,), {})

    constants_module = _module("data.constants")
    constants_module.TOKEN_PAD = 0
    constants_module.TOKEN_END = 1
    constants_module.TOKEN_START = 2
    constants_module.TOKEN_BLANK = 3
    constants_module.VOCAB_SIZE = 1024
    constants_module.sm_tokenizer = types.SimpleNamespace(token_type_list=[])

    config_utils_module = _module("config.utils")
    config_utils_module.DictToObject = lambda value: types.SimpleNamespace(**value)

    log_memory_module = _module("utils.log_memory_usage")
    log_memory_module.profile_cuda_memory = lambda fn: fn


def _install_train_import_stubs():
    t5_module = _module("model.T5")
    t5_module.Transformer = type("Transformer", (nn.Module,), {})

    _module("torchvision")

    gpt_module = _module("model.GPT")
    gpt_module.GPT = type("GPT", (nn.Module,), {})

    omegaconf_module = _module("omegaconf")
    omegaconf_module.OmegaConf = type("OmegaConf", (), {})

    hydra_module = _module("hydra")
    hydra_module.main = lambda *args, **kwargs: (lambda fn: fn)

    dataset_module = _module("data.dataset_Audio2Midi")
    dataset_module.Audio2Midi_Dataset = type("Audio2Midi_Dataset", (), {})

    pl_module = _module("pytorch_lightning")
    pl_module.LightningModule = nn.Module
    pl_module.Trainer = type("Trainer", (), {})

    _module("pytorch_lightning.utilities")
    loggers_module = _module("pytorch_lightning.loggers")
    loggers_module.WandbLogger = type("WandbLogger", (), {})
    loggers_module.TensorBoardLogger = type("TensorBoardLogger", (), {})

    rank_zero_module = _module("pytorch_lightning.utilities.rank_zero")
    rank_zero_module.rank_zero_only = lambda fn: fn

    _module("pytorch_lightning.trainer")
    trainer_states_module = _module("pytorch_lightning.trainer.states")
    trainer_states_module.RunningStage = type("RunningStage", (), {})
    trainer_states_module.TrainerFn = type("TrainerFn", (), {})

    strategies_module = _module("pytorch_lightning.strategies")
    strategies_module.DDPStrategy = type("DDPStrategy", (), {})

    combined_loader_module = _module("pytorch_lightning.utilities.combined_loader")
    combined_loader_module.CombinedLoader = type("CombinedLoader", (), {})

    _module("wandb")

    mel_module = _module("data.mel")
    mel_module.MelSpectrogram = type("MelSpectrogram", (), {})

    spec_augment_module = _module("data.spec_augment")
    spec_augment_module.SpecAugment = type("SpecAugment", (), {})

    _module("PIL")
    pil_image_module = _module("PIL.Image")
    sys.modules["PIL"].Image = pil_image_module
    _module("torchaudio")
    _module("metrics.transcription_metrics")

    _module("sklearn")
    sklearn_metrics_module = _module("sklearn.metrics")
    sklearn_metrics_module.accuracy_score = lambda *args, **kwargs: 0.0

    matplotlib_module = _module("matplotlib")
    matplotlib_pyplot_module = _module("matplotlib.pyplot")
    matplotlib_module.pyplot = matplotlib_pyplot_module

    mir_eval_module = _module("mir_eval")
    mir_eval_util_module = _module("mir_eval.util")
    mir_eval_multipitch_module = _module("mir_eval.multipitch")
    mir_eval_transcription_module = _module("mir_eval.transcription")
    mir_eval_transcription_velocity_module = _module("mir_eval.transcription_velocity")
    mir_eval_util_module.midi_to_hz = lambda value: value
    mir_eval_multipitch_module.evaluate = lambda *args, **kwargs: {}
    mir_eval_transcription_module.precision_recall_f1_overlap = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)
    mir_eval_transcription_velocity_module.precision_recall_f1_overlap = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)
    mir_eval_module.util = mir_eval_util_module
    mir_eval_module.multipitch = mir_eval_multipitch_module
    mir_eval_module.transcription = mir_eval_transcription_module
    mir_eval_module.transcription_velocity = mir_eval_transcription_velocity_module

    symusic_module = _module("symusic")
    symusic_module.Score = type("Score", (), {})
    symusic_module.TimeUnit = types.SimpleNamespace(second="second")

    visualize_module = _module("visualize")
    visualize_module.__path__ = []
    _module("visualize.transcription_visualizer")

    utils_module = _module("utils")
    utils_module.__path__ = []
    _module("utils.sequence_processing")

    tensorboard_module = _module("torch.utils.tensorboard")
    tensorboard_module.SummaryWriter = type("SummaryWriter", (), {})


T5_STUB_MODULE_NAMES = (
    "model.Encoder",
    "model.Decoder",
    "model.Layers",
    "model.Mask",
    "model.HPPNet",
    "data.constants",
    "config.utils",
    "utils.log_memory_usage",
    "_t5_temporal_conv_head_for_test",
)

TRAIN_STUB_MODULE_NAMES = T5_STUB_MODULE_NAMES + (
    "model.T5",
    "model.GPT",
    "torchvision",
    "omegaconf",
    "hydra",
    "data.dataset_Audio2Midi",
    "pytorch_lightning",
    "pytorch_lightning.utilities",
    "pytorch_lightning.loggers",
    "pytorch_lightning.utilities.rank_zero",
    "pytorch_lightning.trainer",
    "pytorch_lightning.trainer.states",
    "pytorch_lightning.strategies",
    "pytorch_lightning.utilities.combined_loader",
    "wandb",
    "data.mel",
    "data.spec_augment",
    "PIL",
    "PIL.Image",
    "torchaudio",
    "metrics.transcription_metrics",
    "sklearn",
    "sklearn.metrics",
    "matplotlib",
    "matplotlib.pyplot",
    "mir_eval",
    "mir_eval.util",
    "mir_eval.multipitch",
    "mir_eval.transcription",
    "mir_eval.transcription_velocity",
    "symusic",
    "visualize",
    "visualize.transcription_visualizer",
    "utils",
    "utils.sequence_processing",
    "torch.utils.tensorboard",
    "_train_checkpoint_helper_for_test",
)


def _load_temporal_conv_pedal_head():
    with _preserve_modules(T5_STUB_MODULE_NAMES):
        _install_t5_import_stubs()
        spec = importlib.util.spec_from_file_location(
            "_t5_temporal_conv_head_for_test",
            PROJECT_DIR / "model" / "T5.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TemporalConvPedalHead


def _load_checkpoint_helpers():
    with _preserve_modules(TRAIN_STUB_MODULE_NAMES):
        _install_t5_import_stubs()
        _install_train_import_stubs()
        spec = importlib.util.spec_from_file_location(
            "_train_checkpoint_helper_for_test",
            PROJECT_DIR / "train.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            module.remove_legacy_linear_pedal_head_layers,
            module.should_load_checkpoint_strictly,
            module.should_resume_full_lightning_checkpoint,
        )


TemporalConvPedalHead = _load_temporal_conv_pedal_head()


def test_temporal_conv_pedal_head_preserves_time_and_backpropagates():
    for kernel_size in (7, 11):
        head = TemporalConvPedalHead(emb_dim=512, kernel_size=kernel_size, hidden=64, dropout=0.0)
        z = torch.randn(2, 17, 512, requires_grad=True)

        logits = head(z)
        assert logits.shape == (2, 17)

        logits.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()


def test_temporal_conv_pedal_head_rejects_even_kernel_size():
    try:
        TemporalConvPedalHead(emb_dim=512, kernel_size=8, hidden=64)
    except ValueError as exc:
        assert "kernel_size must be odd" in str(exc)
    else:
        raise AssertionError("Expected even kernel size to raise ValueError.")


def test_three_temporal_conv_pedal_heads_have_expected_parameter_count():
    heads = (
        TemporalConvPedalHead(emb_dim=512, kernel_size=7, hidden=64),
        TemporalConvPedalHead(emb_dim=512, kernel_size=11, hidden=64),
        TemporalConvPedalHead(emb_dim=512, kernel_size=11, hidden=64),
    )

    total_params = sum(param.numel() for head in heads for param in head.parameters())
    assert 110_000 <= total_params <= 120_000


def test_remove_legacy_linear_pedal_head_layers_keeps_backbone_keys():
    remove_legacy_linear_pedal_head_layers, _, _ = _load_checkpoint_helpers()

    state_dict = {
        "encoder.dense.weight": torch.randn(4, 4),
        "pedal_frame_head.weight": torch.randn(1, 512),
        "pedal_frame_head.bias": torch.randn(1),
        "pedal_onset_head.weight": torch.randn(1, 512),
        "pedal_onset_head.bias": torch.randn(1),
        "pedal_offset_head.weight": torch.randn(1, 512),
        "pedal_offset_head.bias": torch.randn(1),
    }

    removed = remove_legacy_linear_pedal_head_layers(state_dict)

    assert sorted(removed) == [
        "pedal_frame_head.bias",
        "pedal_frame_head.weight",
        "pedal_offset_head.bias",
        "pedal_offset_head.weight",
        "pedal_onset_head.bias",
        "pedal_onset_head.weight",
    ]
    assert set(state_dict) == {"encoder.dense.weight"}


def test_legacy_linear_pedal_heads_disable_strict_loading_and_full_resume():
    _, should_load_checkpoint_strictly, should_resume_full_lightning_checkpoint = _load_checkpoint_helpers()

    legacy_layers = ["pedal_frame_head.weight"]

    assert should_load_checkpoint_strictly(True, legacy_layers) is False
    assert should_load_checkpoint_strictly(True, []) is True
    assert should_resume_full_lightning_checkpoint("lightning", [], legacy_layers) is False
    assert should_resume_full_lightning_checkpoint("lightning", [], []) is True


def test_v4_config_names_temporal_conv_pedal_head_hyperparameters():
    config_text = (PROJECT_DIR / "config" / "experiment_T5_V4_HierarchyPool.yaml").read_text()

    assert "notes: \"Efficient_Transformer_V4_Pedal_TemporalConvHeads-200k\"" in config_text
    assert "pedal_head_type: temporal_conv" in config_text
    assert "pedal_head_hidden: 64" in config_text
    assert "pedal_head_dropout: 0.1" in config_text
    assert "pedal_frame_head_kernel_size: 7" in config_text
    assert "pedal_onset_head_kernel_size: 11" in config_text
    assert "pedal_offset_head_kernel_size: 11" in config_text
    for legacy_key in (
        "pedal_frame_head.weight",
        "pedal_frame_head.bias",
        "pedal_onset_head.weight",
        "pedal_onset_head.bias",
        "pedal_offset_head.weight",
        "pedal_offset_head.bias",
    ):
        assert legacy_key in config_text
