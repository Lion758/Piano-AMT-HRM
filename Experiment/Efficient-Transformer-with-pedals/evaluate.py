from train import MT3Trainer, detect_checkpoint_format, extract_model_state_dict, remove_ignored_layers, validate_transformer_ffn_activation_compatibility
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
    remove_ignored_layers(state_dict, config.model.checkpoint_ignore_layres)
    
    model.model.load_state_dict(state_dict, strict=config.model.strict_checkpoint)
    
    trainer = pl.Trainer(
        logger=[],
        devices=config.devices, # 1 [1,2, 4, 5, 6,7]
        accelerator=config.accelerator, # "gpu"
        strategy=DDPStrategy(find_unused_parameters=True),
        )

    model.test_output_dir =  "__" + config.model.checkpoint_path + "_test"
    os.makedirs(model.test_output_dir, exist_ok=True)
    trainer.test(model.eval())
    
if __name__ == "__main__":
    my_main()
