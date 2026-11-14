# Reference runs: A100, TPU v5e, TPU v6e

Six configurations per chip, run on Google Colab in November 2026 with the team's original training script, which the
code in this repository reproduces bit for bit (checked on CPU: identical per-epoch losses, and identical test predictions
from every checkpoint below). Equivalent command: `python -m soiltwin.train --opt-level L --batch-size B`; up to 100 epochs, early stopping after 15 epochs without
improvement, data split seed 42, init seed 42.

| chip | software |
|---|---|
| NVIDIA A100-SXM4-40GB | PyTorch 2.11.0 + CUDA 13.0 |
| TPU v5e | PyTorch 2.14.1 + TorchTPU |
| TPU v6e | PyTorch 2.14.1 + TorchTPU (same package versions as v5e) |

* `runs.json`: one record per run: test R² and RMSE, best and final epoch, first-epoch and median steady epoch time,
  training and job wall-clock time, throughput, time until validation R² first reaches 0.90 (`time_to_target_s`), and the
  per-epoch history (`epoch_time_s`, `val_r2`, losses). `config` is `o<level>_b<batch>`.
* `preds/<chip>_ladder_<config>.npz`: test-set predictions in physical units (`targets_phys`, `preds_phys`, `mask`,
  `site_id`, `year`, `test_idx`), computed on CPU from each run's best checkpoint.

Epoch times are wall-clock with a device synchronization before each clock read. "Job wall clock" is the shell `time`
of the whole training process, including start-up and data loading.
