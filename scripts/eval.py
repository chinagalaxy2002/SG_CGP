import os
import sys
import json
import argparse
import torch
import pytorch_lightning as pl
import hydra
from hydra.core.global_hydra import GlobalHydra

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def evaluate(checkpoint_path="checkpoints/best_sg_cgp.pt", model_cfg="sg_detr_dq_cgp", losses_cfg="sg_detr_dq_cgp", device="cuda:0"):
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
        
    print(f"\n=======================================================")
    print(f"Evaluating Model: SG-DETR + DQ-CGP")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"=======================================================")
    
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        overrides = [
            "local=default",
            f"model={model_cfg}",
            f"losses={losses_cfg}",
            "trainer.precision=bf16-mixed",
        ]
        cfg = hydra.compose(config_name="train.yaml", overrides=overrides)
        
        datamodule = hydra.utils.instantiate(cfg.data)
        model = hydra.utils.instantiate(cfg.model.runner)
        
        ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt_data["state_dict"] if "state_dict" in ckpt_data else ckpt_data
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            
        gpu_id = int(device.split(":")[-1]) if "cuda" in device and torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=[gpu_id] if torch.cuda.is_available() else "auto",
            precision="bf16-mixed" if torch.cuda.is_available() else "32",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        
        trainer.test(model=model, datamodule=datamodule)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        
        print("\n==================================================================================")
        print(f"{'Metric':<30} | {'Value (%)':<20}")
        print("----------------------------------------------------------------------------------")
        key_metrics = [
            ('test/MR-mAP-Full_Avg', 'MR-mAP-Full_Avg (Core Main Metric)'),
            ('test/MR-mAP-Full_Avg-COMB', 'MR-mAP-Full_Avg-COMB (WBF Post-Processing Fusion)'),
            ('test/MR-R1-Full_0.5', 'MR-R1-Full_0.5 (Top-1 Coarse Recall)'),
            ('test/MR-R1-Full_0.7', 'MR-R1-Full_0.7 (Top-1 Strict Recall)'),
            ('test/MR-R1-Full_mIoU', 'MR-R1-Full_mIoU (Mean IoU Overlap)'),
            ('test/MR-mAP-Full_0.5', 'MR-mAP-Full_0.5 (IoU@0.5 mAP)'),
            ('test/MR-mAP-Full_0.75', 'MR-mAP-Full_0.75 (IoU@0.75 Strict mAP)'),
            ('test/MR-mAP-Short_Avg', 'MR-mAP-Short_Avg (Short Moments <=10s)'),
            ('test/MR-mAP-Middle_Avg', 'MR-mAP-Middle_Avg (Middle Moments 10-30s)'),
            ('test/MR-mAP-Long_Avg', 'MR-mAP-Long_Avg (Long Moments >30s)'),
            ('test/HL-HIT@1-VeryGood', 'HL-HIT@1-VeryGood (Highlight Top-1 Hit)'),
            ('test/HL-mAP-VeryGood', 'HL-mAP-VeryGood (Highlight mAP)'),
            ('test/MR-R1-Full_0.5-COMB', 'MR-R1-Full_0.5-COMB'),
            ('test/MR-R1-Full_0.7-COMB', 'MR-R1-Full_0.7-COMB'),
        ]
        for tag, desc in key_metrics:
            if tag in metrics:
                print(f"{desc:<50} | {metrics[tag]:.3f}")
        print("==================================================================================")
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SG-DETR + DQ-CGP checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_sg_cgp.pt", help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (e.g. cuda:0)")
    args = parser.parse_args()
    evaluate(checkpoint_path=args.checkpoint, device=args.device)
