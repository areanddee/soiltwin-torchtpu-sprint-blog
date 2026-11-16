"""Train and evaluate the CO2-flux LSTM on CUDA, TorchTPU or CPU, at three optimization levels.

  level 0  reference loop: DataLoader, per-batch .item() for the running losses, metrics in NumPy
  level 1  sync-free loop: losses and metrics stay on the device, read back once per epoch
  level 2  level 1 + the whole dataset resident on the device, shuffled there (no DataLoader)

All levels compute the same thing; only where the work happens changes.

    python -m soiltwin.train --opt-level 0 --batch-size 32
"""
import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:  # TorchTPU registers the "tpu" device on import
    import torch_tpu  # noqa: F401
except ImportError:
    pass

from .data import DATA_PATH, TraceDataset, compute_normalizer, load_traces, random_split
from .model import CO2FluxLSTM

# TorchTPU 2.14 noise: .item() on a TPU tensor triggers a spurious non-leaf ".grad" warning inside torch_tpu (this code
# never reads .grad), and get_device_name() is flagged experimental. Neither affects results.
warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor that is not a leaf")
warnings.filterwarnings("ignore", message=".*get_device_name is experimental")

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 9:
    torch.backends.cudnn.enabled = False   # as in the original trainer: its LSTM ran without cuDNN on Hopper and newer


# =============================================================================
# Devices: the whole TorchTPU port is in this section
# =============================================================================
def tpu_available() -> bool:
    """TorchTPU exposes the TPU through the generic torch.accelerator API."""
    acc = getattr(torch, "accelerator", None)
    return bool(acc and acc.is_available() and acc.current_accelerator().type == "tpu")


def pick_device(name: str = "auto") -> torch.device:
    """auto = cuda, else tpu, else cpu."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "tpu" if tpu_available() else "cpu"
    return torch.device(name)


def device_name(device: torch.device) -> str:
    try:
        if device.type == "cuda":
            return torch.cuda.get_device_name(device)
        if device.type == "tpu":
            return str(torch.tpu.get_device_name(0))
    except Exception:
        pass
    return device.type


def sync(device: torch.device) -> None:
    """Wait for queued device work, so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "tpu":
        torch.accelerator.synchronize()


def to_cpu(obj):
    """Copy tensors to the CPU so a checkpoint loads on any device."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


# =============================================================================
# Configuration, losses, metrics
# =============================================================================
@dataclass
class TrainConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    use_attention: bool = True
    use_scale_shift: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 15            # early stopping: epochs without validation-loss improvement
    min_delta: float = 1e-4
    colon_loss_weight: float = 0.1


def masked_mse_loss(pred, target, mask):
    return ((pred - target) ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)


def colonization_bce_loss(pred_probs, labels, mask):
    bce = F.binary_cross_entropy(pred_probs, labels, reduction='none')
    return (bce * mask).sum() / (mask.sum() + 1e-8)


def compute_metrics(pred, target, mask, normalizer):
    """R^2 and RMSE in physical units, in NumPy (copies the batch to the host)."""
    pred_np, target_np, mask_np = pred.detach().cpu().numpy(), target.detach().cpu().numpy(), mask.detach().cpu().numpy()
    pred_denorm = normalizer.denormalize_target(pred_np)
    target_denorm = normalizer.denormalize_target(target_np)
    sel = mask_np.flatten() > 0.5
    p, t = pred_denorm.flatten()[sel], target_denorm.flatten()[sel]
    r2 = 1 - np.sum((p - t) ** 2) / (np.sum((t - np.mean(t)) ** 2) + 1e-8)
    return {'r2': r2, 'rmse': np.sqrt(np.mean((p - t) ** 2))}


# =============================================================================
# Level 0: reference loop
# =============================================================================
def train_epoch(model, loader, optimizer, config, device):
    model.train()
    total_loss = total_mse = total_colon = 0.0
    n_batches = 0
    for inputs, targets, mask, colon_labels in loader:
        inputs, targets, mask, colon_labels = inputs.to(device), targets.to(device), mask.to(device), colon_labels.to(device)
        optimizer.zero_grad()
        if config.use_attention:
            pred, attn, colon_probs = model(inputs, mask, return_attention=True)
            mse_loss = masked_mse_loss(pred, targets, mask)
            colon_loss = colonization_bce_loss(colon_probs, colon_labels, mask)
            loss = mse_loss + config.colon_loss_weight * colon_loss
            total_colon += colon_loss.item()                # host sync
        else:
            pred = model(inputs, mask)
            mse_loss = loss = masked_mse_loss(pred, targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total_loss += loss.item()                           # host sync
        total_mse += mse_loss.item()                        # host sync
        n_batches += 1
    return {'loss': total_loss / n_batches, 'mse': total_mse / n_batches,
            'colon': total_colon / n_batches if config.use_attention else 0.0}


def validate(model, loader, config, normalizer, device):
    model.eval()
    total_loss = 0.0
    all_metrics = []
    n_batches = 0
    with torch.no_grad():
        for inputs, targets, mask, colon_labels in loader:
            inputs, targets, mask, colon_labels = inputs.to(device), targets.to(device), mask.to(device), colon_labels.to(device)
            if config.use_attention:
                pred, attn, colon_probs = model(inputs, mask, return_attention=True)
                loss = masked_mse_loss(pred, targets, mask) + config.colon_loss_weight * colonization_bce_loss(colon_probs, colon_labels, mask)
            else:
                pred = model(inputs, mask)
                loss = masked_mse_loss(pred, targets, mask)
            total_loss += loss.item()
            all_metrics.append(compute_metrics(pred, targets, mask, normalizer))
            n_batches += 1
    return {'loss': total_loss / n_batches,
            'r2': np.mean([m['r2'] for m in all_metrics]), 'rmse': np.mean([m['rmse'] for m in all_metrics])}


# =============================================================================
# Level 1: sync-free loop
# =============================================================================
def _device_metrics(pred, target, mask, normalizer):
    std, mean = float(normalizer.target_std), float(normalizer.target_mean)
    p = pred.reshape(-1) * std + mean
    t = target.reshape(-1) * std + mean
    m = (mask.reshape(-1) > 0.5).to(p.dtype)
    n = m.sum()
    t_mean = (t * m).sum() / n
    ss_res = ((p - t) ** 2 * m).sum()
    ss_tot = ((t - t_mean) ** 2 * m).sum()
    return 1 - ss_res / (ss_tot + 1e-8), torch.sqrt(ss_res / n)


def _forward_losses(model, inputs, targets, mask, colon_labels, config):
    if config.use_attention:
        pred, _attn, colon_probs = model(inputs, mask, return_attention=True)
        mse = masked_mse_loss(pred, targets, mask)
        colon = colonization_bce_loss(colon_probs, colon_labels, mask)
        loss = mse + config.colon_loss_weight * colon
    else:
        pred = model(inputs, mask)
        mse = loss = masked_mse_loss(pred, targets, mask)
        colon = torch.zeros((), device=loss.device)
    return pred, loss, mse, colon


def train_epoch_fast(model, loader, optimizer, config, device):
    model.train()
    totals = torch.zeros(3, device=device)
    n_batches = 0
    for inputs, targets, mask, colon_labels in loader:
        inputs, targets, mask, colon_labels = inputs.to(device), targets.to(device), mask.to(device), colon_labels.to(device)
        optimizer.zero_grad()
        _pred, loss, mse, colon = _forward_losses(model, inputs, targets, mask, colon_labels, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        totals += torch.stack([loss.detach(), mse.detach(), colon.detach()])
        n_batches += 1
    loss, mse, colon = (totals / n_batches).tolist()        # the epoch's only host sync
    return {"loss": loss, "mse": mse, "colon": colon if config.use_attention else 0.0}


def validate_fast(model, loader, config, normalizer, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for inputs, targets, mask, colon_labels in loader:
            inputs, targets, mask, colon_labels = inputs.to(device), targets.to(device), mask.to(device), colon_labels.to(device)
            pred, loss, _mse, _colon = _forward_losses(model, inputs, targets, mask, colon_labels, config)
            r2, rmse = _device_metrics(pred, targets, mask, normalizer)
            rows.append(torch.stack([loss, r2, rmse]))
    loss, r2, rmse = torch.stack(rows).mean(dim=0).tolist()  # per-batch average, like level 0
    return {"loss": loss, "r2": r2, "rmse": rmse}


# =============================================================================
# Level 2: device-resident data
# =============================================================================
class ResidentLoader:
    """Drop-in DataLoader replacement: all samples stacked on the device once; each epoch is one gather."""

    def __init__(self, dataset, batch_size: int, shuffle: bool, device: torch.device, seed: int = 0):
        items = [dataset[i] for i in range(len(dataset))]
        self.tensors = [torch.stack([it[k] for it in items]).to(device) for k in range(4)]
        self.n = len(items)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)  # CPU generator: same order on any device

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        tensors = self.tensors
        if self.shuffle:
            perm = torch.randperm(self.n, generator=self.generator).to(self.device)
            tensors = [t.index_select(0, perm) for t in tensors]
        for start in range(0, self.n, self.batch_size):
            yield tuple(t[start:start + self.batch_size] for t in tensors)


# =============================================================================
# Training and inference
# =============================================================================
def predict(model, loader, normalizer, device):
    """Inference over a loader -> (targets, predictions, mask) as NumPy arrays in physical units."""
    targets_list, preds_list, masks_list = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets, mask, _colon in loader:
            out = model(inputs.to(device), mask=mask.to(device), return_attention=True)
            preds = out[0] if isinstance(out, tuple) else out
            targets_list.append(targets.cpu().numpy())
            preds_list.append(preds.cpu().numpy())
            masks_list.append(mask.cpu().numpy())
    targets_n = np.concatenate(targets_list, axis=0).squeeze(-1)
    preds_n = np.concatenate(preds_list, axis=0).squeeze(-1)
    return normalizer.denormalize_target(targets_n), normalizer.denormalize_target(preds_n), np.concatenate(masks_list, axis=0)


def make_loaders(data, opt_level: int, batch_size: int, device, seed: int = 42, init_seed: Optional[int] = None):
    """Split, normalize and wrap the traces for one optimization level."""
    inputs_list, targets_list, metadata_list = data
    init_seed = seed if init_seed is None else init_seed
    train_idx, val_idx, test_idx = random_split(len(inputs_list), seed=seed)
    sel = lambda arr, idx: [arr[i] for i in idx]  # noqa: E731
    splits = {k: (sel(inputs_list, i), sel(targets_list, i), sel(metadata_list, i))
              for k, i in [("train", train_idx), ("val", val_idx), ("test", test_idx)]}
    normalizer = compute_normalizer(*splits["train"])
    max_len = max(len(x) for x in inputs_list)
    ds = {k: TraceDataset(v[0], v[1], normalizer, max_len=max_len, metadata_list=v[2]) for k, v in splits.items()}
    if opt_level >= 2:
        loaders = {"train": ResidentLoader(ds["train"], batch_size, shuffle=True, device=device, seed=init_seed),
                   "val": ResidentLoader(ds["val"], batch_size, shuffle=False, device=device),
                   "test": ResidentLoader(ds["test"], batch_size, shuffle=False, device=device)}
    else:
        loaders = {"train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True, num_workers=0),
                   "val": DataLoader(ds["val"], batch_size=batch_size, shuffle=False, num_workers=0),
                   "test": DataLoader(ds["test"], batch_size=batch_size, shuffle=False, num_workers=0)}
    return loaders, normalizer, {"train": train_idx, "val": val_idx, "test": test_idx}


def train(data=None, opt_level: int = 0, batch_size: int = 32, epochs: int = 100, device="auto",
          seed: int = 42, init_seed: Optional[int] = None, log_every: int = 10, config: Optional[TrainConfig] = None):
    """Train one model; returns a dict with results, per-epoch history, the best model, normalizer and test predictions."""
    data = data if data is not None else load_traces(DATA_PATH)
    device = pick_device(device) if isinstance(device, str) else device
    init_seed = seed if init_seed is None else init_seed
    config = config or TrainConfig()
    config.batch_size, config.num_epochs = batch_size, epochs

    loaders, normalizer, idx = make_loaders(data, opt_level, batch_size, device, seed, init_seed)
    run_train, run_val = (train_epoch_fast, validate_fast) if opt_level >= 1 else (train_epoch, validate)

    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    model = CO2FluxLSTM(input_size=6, hidden_size=config.hidden_size, num_layers=config.num_layers,
                        dropout=config.dropout, use_attention=config.use_attention,
                        use_scale_shift=config.use_scale_shift).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_r2": [], "epoch_time_s": []}
    best_val_loss, patience, best_epoch, best_state = float("inf"), 0, 0, None
    sync(device)
    t_train0 = time.perf_counter()
    for epoch in range(config.num_epochs):
        t0 = time.perf_counter()
        train_m = run_train(model, loaders["train"], optimizer, config, device)
        val_m = run_val(model, loaders["val"], config, normalizer, device)
        sync(device)
        history["epoch_time_s"].append(time.perf_counter() - t0)
        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_r2"].append(float(val_m["r2"]))
        if log_every and (epoch % log_every == 0 or epoch == config.num_epochs - 1):
            print(f"  epoch {epoch + 1:3d}: train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}  "
                  f"val_r2={val_m['r2']:.4f}  ({history['epoch_time_s'][-1]:.2f} s)")
        if val_m["loss"] < best_val_loss - config.min_delta:
            best_val_loss, patience, best_epoch = val_m["loss"], 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}   # CPU copy
        else:
            patience += 1
            if patience >= config.patience:
                if log_every:
                    print(f"  early stopping at epoch {epoch + 1}")
                break
    train_wall_s = time.perf_counter() - t_train0

    model.load_state_dict(best_state)
    sync(device)
    t0 = time.perf_counter()
    test_m = run_val(model, loaders["test"], config, normalizer, device)
    targets, preds, mask = predict(model, loaders["test"], normalizer, device)
    test_s = time.perf_counter() - t0
    times = history["epoch_time_s"]
    results = {
        "device_name": device_name(device), "torch_version": torch.__version__,
        "opt_level": opt_level, "batch_size": batch_size, "init_seed": init_seed,
        "test_r2": float(test_m["r2"]), "test_rmse": float(test_m["rmse"]),
        "best_epoch": best_epoch + 1, "epochs_run": len(times),
        "first_epoch_s": times[0], "steady_epoch_s": float(np.median(times[1:])) if len(times) > 1 else float("nan"),
        "train_wall_s": train_wall_s, "test_eval_s": test_s,
    }
    return {"results": results, "history": history, "model": model, "normalizer": normalizer, "device": device,
            "test": {"targets": targets, "preds": preds, "mask": mask, "idx": idx["test"],
                     "site_id": np.array([data[2][i]["site_id"] for i in idx["test"]]),
                     "year": np.array([data[2][i]["year"] for i in idx["test"]])}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--opt-level", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "tpu", "cpu"])
    ap.add_argument("--seed", type=int, default=42, help="data split seed")
    ap.add_argument("--init-seed", type=int, default=None, help="weight-init/shuffle seed (default: --seed)")
    args = ap.parse_args()
    out = train(load_traces(args.data), args.opt_level, args.batch_size, args.epochs, args.device, args.seed, args.init_seed)
    r = out["results"]
    print(f"{r['device_name']}: test R2 {r['test_r2']:.4f}  RMSE {r['test_rmse']:.4f}  best epoch {r['best_epoch']}/{r['epochs_run']}  "
          f"first epoch {r['first_epoch_s']:.2f} s  steady epoch {r['steady_epoch_s']:.3f} s")


if __name__ == "__main__":
    main()
