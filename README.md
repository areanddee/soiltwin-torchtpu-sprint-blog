# SoilTwin × TorchTPU: porting a PyTorch LSTM to TPU

<a href="https://colab.research.google.com/github/areanddee/soiltwin-torchtpu-sprint-blog/blob/main/notebooks/soiltwin_torchtpu.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Companion code for our TorchTPU sprint blog post. An LSTM that predicts soil CO₂ flux from soil temperature,
soil moisture and ERA5 weather, trained with unchanged PyTorch code on an NVIDIA A100 and on Google TPU v5e and v6e
through TorchTPU (`torch_tpu`, device `tpu`).

**Start with the notebook**, [`notebooks/soiltwin_torchtpu.ipynb`](notebooks/soiltwin_torchtpu.ipynb). In Colab, choose a
TPU (v5e-1 or v6e-1) or GPU runtime, then *Run all*. Part 1 trains and tests the model on your accelerator in about
5 minutes and compares it with ours; part 2 walks through our results on three chips.

## What's here

| path | contents |
|---|---|
| `soiltwin/model.py` | the network: 2-layer LSTM, temporal attention, colonization detector, scale-shift head (~70k parameters) |
| `soiltwin/data.py` | loading, the fixed random split, normalization, the PyTorch `Dataset` |
| `soiltwin/train.py` | device handling (CUDA / TorchTPU / CPU), training at optimization levels 0–2, inference |
| `data/soil_co2_traces.npz` | 300 traces × 2016 half-hour steps; see [`data/README.md`](data/README.md) |
| `results/` | our reference runs on A100, TPU v5e and TPU v6e; see [`results/README.md`](results/README.md) |
| `figures/` | the plots used in the notebook |

## Command line

```bash
pip install -r requirements.txt          # torch, numpy (+ matplotlib, pandas for the figures)
python -m soiltwin.train --opt-level 0 --batch-size 32      # naive port
python -m soiltwin.train --opt-level 2 --batch-size 128     # device-resident data, bigger batch
```

`--device auto` (the default) picks CUDA, then TPU, then CPU. On a TPU runtime TorchTPU is preinstalled by Colab.

Optimization levels: **0** is the original training loop (a `DataLoader`, per-batch `.item()` calls); **1** keeps losses
and metrics on the device and syncs once per epoch; **2** also keeps the whole dataset on the device. All three compute
the same model; on CPU they reproduce each other bit for bit.

## Licence and data attribution

Code: [Apache-2.0](LICENSE). The training data combine synthetic site data with ERA5 reanalysis: *contains modified
Copernicus Climate Change Service information (2026).* Hersbach, H. et al. (2020), The ERA5 global reanalysis,
*Q. J. R. Meteorol. Soc.* 146, 1999–2049, doi:10.1002/qj.3803.
