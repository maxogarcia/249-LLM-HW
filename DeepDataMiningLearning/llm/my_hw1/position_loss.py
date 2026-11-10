"""Part 3 (deeper): loss at each position inside the context window, for the T=512 models.

If the per-position curve is flat after position ~64, the model gets nothing from history further back,
and the T=256→512 gain would just be fewer "cold-start" characters per window.

Run from the llm folder:  python my_hw1/position_loss.py
Writes my_hw1/position_loss.png and my_hw1/position_loss.md.
"""
import math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from nanolm.generate import load_run
from nanolm.data import TokenData

SWEEP = "outputs/nanolm/sweeps/my_context"
T = 512
SEEDS = [1337, 42]
MODELS = {"gpt_modern": "Transformer (modern)", "lstm": "LSTM 2×256"}
STRIDE = 16
BUCKETS =[(0, 8), (8, 64), (64, 128), (128, 256), (256, 512)]
LN2 = math.log(2)                     # char tokenizer: 1 byte per token, so bpb = loss / ln 2

@torch.no_grad()
def per_position_bpb(run_dir):
    cfg, model, tok, dinfo, meta = load_run(run_dir)
    model.float()                     # full precision for the measurement
    # Overlapping windows (a new one every STRIDE chars): each position is then averaged over ~all of the
    # validation text, not just the 217 chars that happen to sit at that offset in non-overlapping windows.
    val = torch.from_numpy(np.asarray(TokenData(meta).val, dtype=np.int64))
    starts = torch.arange(0, len(val) - T - 1, STRIDE)
    idx = starts[:, None] + torch.arange(T + 1)[None, :]
    win = val[idx]
    x, y = win[:, :-1], win[:, 1:]
    n_win = len(starts)
    total = torch.zeros(T, dtype=torch.float64)
    for i in range(0, n_win, 32):
        xb, yb = x[i:i + 32].to(dinfo.device), y[i:i + 32].to(dinfo.device)
        logits = model(xb).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="none")
        total += loss.view(-1, T).sum(0).double().cpu()
    return (total / n_win / LN2).numpy(), n_win

curves = {}
for arch in MODELS:
    per_seed = []
    for s in SEEDS:
        c, n_win = per_position_bpb(f"{SWEEP}/{arch}_T{T}_s{s}")
        per_seed.append(c)
        print(f"{arch}_T{T}_s{s}: {n_win} windows, mean bpb {c.mean():.4f}")
    curves[arch] = np.mean(per_seed, axis=0)

# ---- plot: raw per-position bpb (faint) + smoothed (bold), log-scaled position axis ----
def smooth(c, k=16):
    return np.convolve(c, np.ones(k) / k, mode="valid"), np.arange(k // 2, len(c) - k // 2 + 1)

fig, ax = plt.subplots(figsize=(8, 4.8))
for arch, label in MODELS.items():
    c = curves[arch]
    line, = ax.plot(np.arange(1, T + 1), c, alpha=0.25, linewidth=0.8)
    s, xs = smooth(c)
    ax.plot(xs + 1, s, color=line.get_color(), linewidth=2, label=f"{label}, T={T} (16-pos moving avg)")
ax.axvline(64, color="gray", linestyle=":", linewidth=1)
ax.set_xscale("log", base=2)
ax.set_ylim(1.9, 3.0)
ax.text(68, 1.92, "position 64", color="gray", va="bottom", fontsize=8)
ax.set_xlabel("position inside the window (characters of history + 1)")
ax.set_ylabel("validation bpb at that position")
ax.set_title(f"Per-position loss, T={T} models (mean over {len(SEEDS)} seeds)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("my_hw1/position_loss.png", dpi=150)

# ---- bucket table ----
head = ["position range"] + [MODELS[a] for a in MODELS]
L = [f"**Table 5. Mean validation bpb by position inside the window, T={T} models, mean over seeds "
     f"{SEEDS[0]} and {SEEDS[1]}; overlapping windows every {STRIDE} characters.**\n", "| " + " | ".join(head) + " |", "|---|" + "---:|" * len(MODELS)]
for lo, hi in BUCKETS:
    L.append(f"| {lo}–{hi - 1} | " + " | ".join(f"{curves[a][lo:hi].mean():.4f}" for a in MODELS) + " |")
L.append("| all 0–511 | " + " | ".join(f"{curves[a].mean():.4f}" for a in MODELS) + " |")
Path("my_hw1/position_loss.md").write_text("\n".join(L) + "\n", encoding="utf-8")
np.savetxt("my_hw1/position_loss.csv", np.column_stack([np.arange(T)] + [curves[a] for a in MODELS]),
           delimiter=",", header="position," + ",".join(MODELS), comments="", fmt="%.6f")
print("\n".join(L))
