"""Load the soil CO2 traces, split them, normalize them, and serve them as (inputs, target, mask, colonization) tensors.

data/soil_co2_traces.npz holds 300 traces x 2016 half-hour steps (6 weeks):
  inputs   (300, 2016, 5) float32  T_soil, theta_soil, SSRD, T_air, day-of-year/366
  targets  (300, 2016, 3) float32  MBC, SOC, CO2_flux_heterotrophic (the model predicts the last one)
  initial_mbc, colonization_lag_days (300,) per-trace scalars; site_id, year (300,) labels
The model sees 6 channels: the 5 above plus initial MBC broadcast over time.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "soil_co2_traces.npz")
CHANNELS = ['T_soil', 'theta_soil', 'SSRD', 'T_air', 'doy', 'initial_MBC']
N_CHANNELS = len(CHANNELS)
DT_HOURS = 0.5


def load_traces(path: str = DATA_PATH):
    """Return (inputs_list, targets_list, metadata_list), one entry per trace."""
    d = np.load(path)
    inputs_list = list(d["inputs"])
    targets_list = list(d["targets"])
    metadata_list = [{"day_1_MBC": float(d["initial_mbc"][i]),
                      "colonization_lag_days": float(d["colonization_lag_days"][i]),
                      "site_id": int(d["site_id"][i]), "year": int(d["year"][i])}
                     for i in range(len(inputs_list))]
    return inputs_list, targets_list, metadata_list


def random_split(n: int, val_frac: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded random split: 10% test, 10% validation, the rest training. Returns index arrays."""
    order = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(round(n * val_frac)))
    n_val = max(1, int(round(n * val_frac)))
    return order[n_test + n_val:], order[n_test:n_test + n_val], order[:n_test]


@dataclass
class Normalizer:
    """Per-channel input standardization and target standardization, fitted on the training split."""
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: float
    target_std: float

    def normalize_inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.input_mean) / (self.input_std + 1e-8)

    def normalize_target(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / (self.target_std + 1e-8)

    def denormalize_target(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.target_std + self.target_mean


def compute_normalizer(inputs_list: list, targets_list: list, metadata_list: list) -> Normalizer:
    all_inputs = np.concatenate(inputs_list, axis=0)
    all_co2 = np.concatenate([t[:, 2] for t in targets_list], axis=0)
    n_existing = all_inputs.shape[1]
    input_mean = np.zeros(N_CHANNELS, dtype=np.float32)
    input_std = np.ones(N_CHANNELS, dtype=np.float32)
    input_mean[:n_existing] = np.mean(all_inputs, axis=0).astype(np.float32)
    input_std[:n_existing] = np.std(all_inputs, axis=0).astype(np.float32)
    initial_mbcs = np.array([m.get('day_1_MBC', 0.0) for m in metadata_list])
    input_mean[-1] = float(np.mean(initial_mbcs))
    input_std[-1] = float(np.std(initial_mbcs)) if np.std(initial_mbcs) > 0 else 1.0
    return Normalizer(input_mean=input_mean, input_std=input_std,
                      target_mean=float(np.mean(all_co2)), target_std=float(np.std(all_co2)))


class TraceDataset(Dataset):
    """Item i -> (inputs (T, 6), CO2 flux target (T, 1), mask (T,), colonization label (T,)), normalized."""

    def __init__(self, inputs_list: List[np.ndarray], targets_list: List[np.ndarray], normalizer: Normalizer,
                 max_len: Optional[int] = None, metadata_list: Optional[List[dict]] = None):
        self.inputs_list = inputs_list
        self.targets_list = targets_list
        self.normalizer = normalizer
        self.max_len = max_len or max(len(x) for x in inputs_list)
        self.metadata_list = metadata_list

    def __len__(self) -> int:
        return len(self.inputs_list)

    def __getitem__(self, idx: int):
        inputs = np.array(self.inputs_list[idx], dtype=np.float32)
        targets = np.array(self.targets_list[idx], dtype=np.float32)
        co2_flux = targets[:, 2:3]
        seq_len = min(len(inputs), self.max_len)
        meta = self.metadata_list[idx] if self.metadata_list is not None else {}

        inputs_full = np.zeros((seq_len, N_CHANNELS), dtype=np.float32)
        n_existing = inputs.shape[1]
        inputs_full[:, :n_existing] = inputs[:seq_len]
        inputs_full[:, -1] = meta.get('day_1_MBC', 0.0)        # initial MBC as a constant channel

        inputs_padded = np.zeros((self.max_len, N_CHANNELS), dtype=np.float32)
        targets_padded = np.zeros((self.max_len, 1), dtype=np.float32)
        mask = np.zeros(self.max_len, dtype=np.float32)
        colon_labels = np.zeros(self.max_len, dtype=np.float32)
        inputs_padded[:seq_len] = self.normalizer.normalize_inputs(inputs_full)
        targets_padded[:seq_len] = self.normalizer.normalize_target(co2_flux[:seq_len])
        mask[:seq_len] = 1.0
        if self.metadata_list is not None:                       # 1 once colonization has begun
            time_hours = np.arange(seq_len) * DT_HOURS
            colon_labels[:seq_len] = (time_hours > meta.get('colonization_lag_days', 10.0) * 24.0).astype(np.float32)

        return (torch.from_numpy(inputs_padded), torch.from_numpy(targets_padded),
                torch.from_numpy(mask), torch.from_numpy(colon_labels))
