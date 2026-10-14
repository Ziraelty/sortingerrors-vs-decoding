# Simulated spike-sorting errors vs. decoding performance
# NLB MC_Maze (jenkins_maze_train) with torch_brain.
# Dataset class and GRU adapted from the torch_brain NLB Maze tutorial.

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torch_brain.data import Interval
from torch_brain.datasets import DatasetIndex, PeiPandarinathNLB2021
from torch_brain.samplers import TrialSampler
from torch_brain.utils import bin_spikes

BIN_SIZE = 0.01
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
DATA_ROOT = "data/processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


class SimpleNLBMazeDataset(PeiPandarinathNLB2021):
    sample_length = 0.7
    out_dim = 2
    out_sampling_rate = 1000.0

    def __init__(self, root, split: Literal["train", "val"], bin_size):
        super().__init__(root=root, recording_ids=["jenkins_maze_train"])
        assert split in ("train", "val")
        self.split = split
        self.bin_size = bin_size
        self.out_samples = round(self.sample_length * self.out_sampling_rate)
        self.num_bins = round(self.sample_length / self.bin_size)
        self.num_units = len(self.get_unit_ids())

    def get_sampling_intervals(self, *args, **kwargs):
        rid = self.recording_ids[0]
        rec = self.get_recording(rid)
        onset = rec.trials.move_onset_time
        trials = Interval(onset - 0.25, onset + 0.45)
        split = rec.trials.split_indicator.astype(str)
        return {rid: trials.select_by_mask(split == self.split)}

    def __getitem__(self, index: DatasetIndex):
        data = super().__getitem__(index)
        X = bin_spikes(data.spikes, num_units=len(data.units), bin_size=self.bin_size)
        Y = data.hand.vel / 200.0
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


train_ds = SimpleNLBMazeDataset(DATA_ROOT, "train", BIN_SIZE)
val_ds = SimpleNLBMazeDataset(DATA_ROOT, "val", BIN_SIZE)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    sampler=TrialSampler(sampling_intervals=train_ds.get_sampling_intervals(), shuffle=True),
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    sampler=TrialSampler(sampling_intervals=val_ds.get_sampling_intervals()),
)
print(train_ds.num_units, "units,", train_ds.num_bins, "bins per trial")


class GRU(nn.Module):
    def __init__(self, in_units, out_dim, out_samples, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.out_samples = out_samples
        self.gru = nn.GRU(in_units, hidden_dim, num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.readout = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.readout(self.gru(x)[0]).permute(0, 2, 1)
        y = nn.functional.interpolate(y, self.out_samples, mode="linear")
        return y.permute(0, 2, 1)


model = GRU(train_ds.num_units, train_ds.out_dim, train_ds.out_samples).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=LR)

for _ in tqdm(range(EPOCHS)):
    model.train()
    for X, Y in train_loader:
        loss = nn.functional.mse_loss(model(X.to(device)), Y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
model.eval()


def corrupt(X, kind, level, rng, mean_rate):
    # X: spike counts (trials, bins, units), same units affected in every trial
    X = X.copy()
    n = X.shape[-1]
    idx = rng.choice(n, size=int(round(level * n)), replace=False)

    if kind == "dropout":
        X[..., idx] = 0
    elif kind == "missed":
        X = rng.binomial(X, 1 - level)
    elif kind == "split":
        # half of the spikes go to a new unit the decoder has never seen
        X[..., idx] = rng.binomial(X[..., idx], 0.5)
    elif kind == "merge":
        half = len(idx) // 2
        a, b = idx[:half], idx[half:2 * half]
        X[..., a] += X[..., b]
        X[..., b] = 0
    elif kind == "false_pos":
        X += rng.poisson(level * mean_rate, X.shape)
    elif kind == "swap":
        X[..., idx] = X[..., rng.permutation(idx)]
    return X


Xs, Ys = zip(*[(X, Y) for X, Y in val_loader])
Xv = torch.cat(Xs).numpy().round().astype(int)
Yv = torch.cat(Ys)
mean_rate = Xv.mean()


def r2_under(kind, level, seed):
    Xc = corrupt(Xv, kind, level, np.random.default_rng(seed), mean_rate)
    with torch.no_grad():
        pred = model(torch.from_numpy(Xc).float().to(device)).cpu()
    return r2_score(Yv.flatten(0, 1), pred.flatten(0, 1))


print(f"clean R2: {r2_under('dropout', 0, 0):.3f}")

kinds = ["dropout", "missed", "split", "merge", "false_pos", "swap"]
levels = [0, 0.1, 0.25, 0.5]
rows = []
for k in kinds:
    for l in levels:
        for s in range(5):
            rows.append((k, l, s, r2_under(k, l, s)))

df = pd.DataFrame(rows, columns=["kind", "level", "seed", "r2"])
df.to_csv("results.csv", index=False)
summary = df.groupby(["kind", "level"]).r2.agg(["mean", "std"]).reset_index()
print(summary.pivot(index="kind", columns="level", values="mean").round(3))

labels = {
    "dropout": "Dropout (units lost)",
    "missed": "Missed spikes",
    "split": "Split units",
    "merge": "Merged units",
    "false_pos": "False-positive spikes",
    "swap": "Swapped labels",
}
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

fig, ax = plt.subplots(figsize=(7, 4.5))
for kind, color in zip(kinds, colors):
    g = summary[summary.kind == kind]
    ax.errorbar(g.level * 100, g["mean"], yerr=g["std"], color=color, lw=2,
                marker="o", ms=6, capsize=3, label=labels[kind])
ax.axhline(summary.loc[summary.level == 0, "mean"].mean(), ls="--", color="gray", lw=1,
           label="Clean data")
ax.set_xticks([l * 100 for l in levels])
ax.set(xlabel="Corruption level (%)", ylabel="Validation R² (hand velocity)",
       title="Which spike-sorting errors hurt decoding?")
ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("sorting_errors.png", dpi=200)
plt.show()
