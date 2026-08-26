# Improved Baseline -> DQ-CGP fine-tuning

This directory is an isolated experiment overlay. It does not modify the
existing Baseline or DQ-CGP implementation.

Implemented changes:

- Restores the exact Baseline `LocalSaliencyHead` architecture.
- Rejects checkpoint mismatches outside newly introduced `query_cgp.*` keys.
- Uses separate LRs: inherited base `5e-5`, inherited saliency `2e-5`, and
  Query-CGP `2e-4`.
- Freezes inherited parameters for the first 2 epochs.
- Holds beta at `0.005` for 3 epochs, then ramps to `0.02` by epoch 10.
- Starts DQ losses at bind `0.1` and route `0.001`; decays them by `0.2` when
  route loss reaches `-1.8` or validation stalls for 3 epochs.
- Uses 3 warmup epochs, cosine decay, at most 40 epochs, early-stopping
  patience 10, top-3 checkpoints, and a last checkpoint.

Configuration-only dry run:

```bash
DRY_RUN=1 TARGET_GPU=0 bash code/dq_cgp_ft_improved/run_train.sh
```

Actual run:

```bash
TARGET_GPU=0 bash code/dq_cgp_ft_improved/run_train.sh
```

Additional Hydra overrides can be appended to the command. For example:

```bash
TARGET_GPU=0 bash code/dq_cgp_ft_improved/run_train.sh \
  ++model.runner.query_cgp_lr=3e-4 \
  ++callbacks.dq_finetune_control.beta_end=0.03
```
