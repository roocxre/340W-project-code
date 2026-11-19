"""Stage 6: EEGNet on the raw windows, the end-to-end candidate.

    caffeinate -i ../synthesis/.venv/bin/python scripts/06_eegnet.py [--task bin] [--seeds 3]

Input per window: the 26-channel, 4 s, 200 Hz window behind the CSV row, projected onto the
25-dimensional sum-zero subspace (the data are average-referenced, so this loses nothing),
Euclidean-aligned per subject (whitened by the subject's mean covariance, no labels), and
decimated to 100 Hz (the data are low-passed at 40 Hz): shape (25, 400).

EEGNet-8,2 (Lawhern et al. 2018): temporal convolution (0.5 s kernel), depthwise spatial
convolution, separable convolution, two average-pooling stages, dropout 0.25, max-norm
constraints. Adam (lr 1e-3), batch 64, class-balanced cross-entropy, at most 80 epochs.
The epoch with the lowest validation loss is kept (early stopping on the 6 val subjects,
patience 15). Each seed is a separate network; their test probabilities are averaged.

Outputs: cache/models/<task>_eegnet.json, results/predictions/<task>__eegnet.npz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import decimate
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbackml as ml  # noqa: E402
import nbacksplit as ns  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
FS = 100


def aligned_windows(df):
    """(n, 25, 400) float32: sum-zero projection, per-subject Euclidean alignment, 100 Hz."""
    from pyriemann.geometry.base import invsqrtm
    B = ml._sumzero_basis(26)
    out = np.empty((len(df), 25, 400), dtype=np.float32)
    for subject, idx in df.groupby("subject").indices.items():
        x = ns.windows(df.iloc[idx]).astype(float)
        x = np.einsum("cd,ncs->nds", B, x)
        x = decimate(x, 2, ftype="fir", zero_phase=True, axis=-1)
        R = np.einsum("ncs,nds->cd", x, x) / (x.shape[0] * x.shape[2])
        out[idx] = np.einsum("cd,nds->ncs", invsqrtm(R), x)
    return out


class MaxNormLinear(nn.Linear):
    def __init__(self, *a, max_norm=0.25, **k):
        super().__init__(*a, **k)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            self.weight.data = torch.renorm(self.weight.data, 2, 0, self.max_norm)
        return super().forward(x)


class MaxNormConv(nn.Conv2d):
    def __init__(self, *a, max_norm=1.0, **k):
        super().__init__(*a, **k)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            self.weight.data = torch.renorm(self.weight.data, 2, 0, self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(self, n_ch, n_t, n_cls, F1=8, D=2, F2=16, k_t=FS // 2, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, k_t), padding=(0, k_t // 2), bias=False),
            nn.BatchNorm2d(F1),
            MaxNormConv(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop),
            nn.Flatten())
        with torch.no_grad():
            n_flat = self.net(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.head = MaxNormLinear(n_flat, n_cls)

    def forward(self, x):
        return self.head(self.net(x.unsqueeze(1)))


def predict(model, X, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + bs]).to(DEV)), 1).cpu().numpy())
    return np.vstack(out)


def train_one(X, y, Xv, yv, k, seed, epochs=80, patience=15, bs=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EEGNet(X.shape[1], X.shape[2], k).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w = torch.tensor(len(y) / (k * np.bincount(y, minlength=k)), dtype=torch.float32, device=DEV)
    lossf = nn.CrossEntropyLoss(weight=w)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(y))
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(torch.from_numpy(X[b]).to(DEV)), torch.from_numpy(y[b]).to(DEV))
            loss.backward()
            opt.step()
        pv = predict(model, Xv)
        vloss = float(nn.functional.nll_loss(torch.log(torch.from_numpy(pv) + 1e-9), torch.from_numpy(yv),
                                             weight=w.cpu()))
        hist.append(dict(epoch=ep + 1, val_loss=round(vloss, 4), val_acc=round(100 * float((pv.argmax(1) == yv).mean()), 2)))
        if vloss < best - 1e-4:
            best, wait = vloss, 0
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def run(task, seeds):
    out = ml.CACHE / "models" / f"{task}_eegnet.json"
    if out.exists():
        return json.loads(out.read_text())
    t0 = time.time()
    d = {s: ml.frame(s, task) for s in ("train", "val", "test")}
    y = {s: ml.labels(d[s], task) for s in d}
    X = {s: aligned_windows(d[s]) for s in d}
    k = len(ml.TASKS[task]["levels"])
    P = {s: np.zeros((len(y[s]), k)) for s in ("val", "test")}
    hists = []
    for seed in range(seeds):
        model, hist = train_one(X["train"], y["train"], X["val"], y["val"], k, seed)
        best_ep = min(hist, key=lambda h: h["val_loss"])
        hists.append(dict(seed=seed, best_epoch=best_ep["epoch"], epochs_run=len(hist), history=hist))
        for s in P:
            P[s] += predict(model, X[s]) / seeds
        print(f"  {task} seed {seed}: best epoch {best_ep['epoch']} of {len(hist)}, val acc {best_ep['val_acc']:.1f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    r = dict(model="eegnet", task=task, device=str(DEV), seeds=seeds, val=ml.metrics(y["val"], P["val"]),
             test=ml.metrics(y["test"], P["test"]), seconds=round(time.time() - t0), training=hists)
    ml.save_predictions(task, "eegnet", **{f"{s}_proba": P[s] for s in P}, **{f"{s}_y": y[s] for s in P},
                        **{f"{s}_row": d[s].row_id.to_numpy() for s in P})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="*", default=list(ml.TASKS))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    for task in a.task:
        r = run(task, a.seeds)
        print(f"{task:6s} eegnet   val acc {r['val']['acc']:.1f} | test acc {r['test']['acc']:.1f} "
              f"bacc {r['test']['bacc']:.1f} kappa {r['test']['kappa']:.1f} auc {r['test']['auc']:.1f} "
              f"({r['seconds']} s, {r['device']})", flush=True)


if __name__ == "__main__":
    main()
