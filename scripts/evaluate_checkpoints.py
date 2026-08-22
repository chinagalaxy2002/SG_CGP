import os
import sys
import json
import torch
import pytorch_lightning as pl
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Add project root to sys.path
sys.path.insert(0, '/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr')

def evaluate_single_checkpoint(ckpt_path, model_cfg_name, losses_cfg_name, device="cuda:0"):
    print(f"\n=======================================================")
    print(f"Evaluating Checkpoint: {ckpt_path}")
    print(f"Model config: {model_cfg_name}, Losses config: {losses_cfg_name}")
    print(f"=======================================================")
    
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        overrides = [
            "local=guoxiangyu",
            f"model={model_cfg_name}",
            f"losses={losses_cfg_name}",
            "trainer.precision=bf16-mixed",
        ]
        cfg = hydra.compose(config_name="train.yaml", overrides=overrides)
        
        # Instantiate datamodule and model
        datamodule = hydra.utils.instantiate(cfg.data)
        model = hydra.utils.instantiate(cfg.model.runner)
        
        # Load state dict
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt_data["state_dict"] if "state_dict" in ckpt_data else ckpt_data
        
        # Load into model
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"State dict loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[int(device.split(":")[-1])],
            precision="bf16-mixed",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        
        results = trainer.test(model=model, datamodule=datamodule)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        return metrics

if __name__ == "__main__":
    checkpoints = [
        {
            "name": "DQ-CGP Best (Epoch 104)",
            "path": "/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/logs/sg_detr_dq_cgp_exp/runs/2026-08-21_16-21-37/checkpoints/epoch_epoch=104.ckpt",
            "model_cfg": "sg_detr_dq_cgp",
            "losses_cfg": "sg_detr_dq_cgp",
            "device": "cuda:0"
        },
        {
            "name": "Baseline Reproduce Best (Epoch 100)",
            "path": "/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/logs/qvhighlights_reproduce/runs/2026-08-21_16-13-56/checkpoints/epoch_epoch=100.ckpt",
            "model_cfg": "default",
            "losses_cfg": "default",
            "device": "cuda:0"
        },
        {
            "name": "Original Official Baseline (best_qvhighlights_2.pt)",
            "path": "/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/best_qvhighlights_2.pt",
            "model_cfg": "default",
            "losses_cfg": "default",
            "device": "cuda:0"
        }
    ]
    
    all_results = {}
    for item in checkpoints:
        if os.path.exists(item["path"]):
            m = evaluate_single_checkpoint(item["path"], item["model_cfg"], item["losses_cfg"], item["device"])
            all_results[item["name"]] = m
        else:
            print(f"File not found: {item['path']}")
            
    # Save results
    out_file = "/home/guoxiangyu/VLMbasedIter_momentretrival/sg-detr/logs/comparison_test_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll results saved to {out_file}")
