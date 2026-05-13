from train import (
    MT3Trainer,
    detect_checkpoint_format,
    extract_model_state_dict,
    remove_ignored_layers,
    remove_legacy_linear_pedal_head_layers,
    should_load_checkpoint_strictly,
    validate_transformer_ffn_activation_compatibility,
)
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
from collections import defaultdict


@hydra.main(config_path="config", config_name="main_config", version_base = None)
def my_main(config: OmegaConf):
    # Create model.
    config.training.mode = "test"
    model = MT3Trainer(config)
    print(model)
    # Load checkpoint.
    checkpoint = torch.load(config.model.checkpoint_path, map_location="cpu")
    checkpoint_format = detect_checkpoint_format(checkpoint)
    state_dict = extract_model_state_dict(checkpoint, checkpoint_format)
    validate_transformer_ffn_activation_compatibility(state_dict, config.model.mlp_activations)
    ignored_layers = remove_ignored_layers(state_dict, config.model.checkpoint_ignore_layres)
    legacy_pedal_head_layers = remove_legacy_linear_pedal_head_layers(state_dict)
    skipped_layers = ignored_layers + legacy_pedal_head_layers
    
    load_strict = should_load_checkpoint_strictly(config.model.strict_checkpoint, skipped_layers)
    model.model.load_state_dict(state_dict, strict=load_strict)
    
    trainer = pl.Trainer(
        logger=[],
        devices=config.devices, # 1 [1,2, 4, 5, 6,7]
        accelerator=config.accelerator, # "gpu"
        strategy=DDPStrategy(find_unused_parameters=True),
        )

    evaluation_config = config.get("evaluation", {})
    evaluation_subset = str(evaluation_config.get("subset", "test")) if evaluation_config else "test"
    valid_evaluation_subsets = {"test", "validation"}
    if evaluation_subset not in valid_evaluation_subsets:
        raise ValueError(
            f"evaluation.subset must be one of {sorted(valid_evaluation_subsets)}, got {evaluation_subset!r}."
        )
    configured_test_output_dir = evaluation_config.get("test_output_dir", None) if evaluation_config else None
    if configured_test_output_dir:
        model.test_output_dir = os.path.expanduser(str(configured_test_output_dir))
    else:
        model.test_output_dir =  "__" + config.model.checkpoint_path + "_test"
    os.makedirs(model.test_output_dir, exist_ok=True)
    trainer.test(model.eval(), dataloaders=model.test_dataloader(subset=evaluation_subset))
    
if __name__ == "__main__":
    my_main()
