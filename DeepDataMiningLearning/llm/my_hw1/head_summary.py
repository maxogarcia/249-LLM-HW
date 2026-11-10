"""Part 4 submission table + figure: every head type we measured, its strongest head in each seed,
the average attention weight on its target for every text, the chance level, and a verdict.

Needs my_hw1/head_stats.csv (run my_hw1/head_stats.py first).
Run from the llm folder:  python my_hw1/head_summary.py
Writes my_hw1/head_types.md / .html / .csv and my_hw1/head_types.png.
"""
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("my_hw1/head_stats.csv")
TEXTS = ["shakespeare", "prose", "code", "multiline"]
TEXT_LABEL = {"shakespeare": "Shakespeare", "prose": "modern prose", "code": "code", "multiline": "multi-line Shakespeare"}
SEEDS = {"s1337": "gpt_modern_s1337", "s42": "gpt_modern_s42"}
TYPES = [  # (metric in head_stats.csv, head type name, what it attends to)
    ("prev",           "previous-character", "t−1"),
    ("prev2",          "two-back",           "t−2"),
    ("word_start_far", "word-start",         "first letter of the current word (4th letter on)"),
    ("line_start",     "line-start",         "most recent newline (2nd char of the line on)"),
    ("first",          "first-position",     "position 0 of the text"),
]
FOUND, WEAK = 3.0, 2.0   # verdict: median weight/chance over texts, in BOTH seeds

rows = []
for metric, tname, target in TYPES:
    for seed in SEEDS:
        sub = df[(df.metric == metric) & (df.model == seed)]
        # strongest head = highest mean weight over the texts where the type can be measured
        best = sub.groupby(["layer", "head"])["weight"].mean().idxmax()
        h = sub[(sub.layer == best[0]) & (sub["head"] == best[1])].set_index("text")
        r = {"type": tname, "metric": metric, "target": target, "seed": seed, "head": f"L{best[0]}H{best[1]}"}
        for t in TEXTS:
            if t in h.index:
                r[f"{t}_w"], r[f"{t}_chance"], r[f"{t}_n"] = h.loc[t, "weight"], h.loc[t, "chance"], h.loc[t, "n_queries"]
            else:
                r[f"{t}_w"] = r[f"{t}_chance"] = r[f"{t}_n"] = np.nan
        ratios = [r[f"{t}_w"] / r[f"{t}_chance"] for t in TEXTS]
        r["median_x_chance"] = np.nanmedian(ratios)
        r["min_x_chance"] = np.nanmin(ratios)
        rows.append(r)
out = pd.DataFrame(rows)
verdict = {}
for tname, g in out.groupby("type"):
    # found: >= 3x chance on EVERY text, in BOTH seeds; weak: median >= 2x in both seeds
    verdict[tname] = ("found" if g["min_x_chance"].min() >= FOUND
                      else "weak" if g["median_x_chance"].min() >= WEAK else "not found")
out["verdict"] = out["type"].map(verdict)
out.to_csv("my_hw1/head_types.csv", index=False)

# ---------------- table ----------------
cap = ("Table 9. Attention-head types in the 4-layer character Transformer (RMSNorm + RoPE + SwiGLU), "
       "two training seeds. For each type, the strongest head in each seed and its average attention weight on "
       "the target position, per text; chance in parentheses = average of 1/(t+1) over the same query positions "
       "(uniform attention). Verdict: found if the weight is ≥ 3× chance on every text in both seeds; weak if the "
       "median over texts is ≥ 2× chance in both seeds; otherwise not found. Word-start and line-start skip queries where the target is t−1 (and "
       "word-start also t−2) so a previous-character or two-back head cannot score. n/a = the text has no "
       "newline. Texts: “ROMEO:⏎But soft! what light through”; “The bus was late again, so I walked to work.”; "
       "“def add(a, b):⏎    return a + b”; a 7-line Shakespeare dialogue (line-start only needs several lines).")
head = (["head type", "attends to", "seed", "head"] + [TEXT_LABEL[t] for t in TEXTS]
        + ["median × chance", "lowest × chance", "verdict"])
body = []
for _, r in out.iterrows():
    cells = [r["type"], r["target"], SEEDS[r["seed"]], r["head"]]
    for t in TEXTS:
        cells.append("n/a" if np.isnan(r[f"{t}_w"]) else f"{r[f'{t}_w']:.2f} ({r[f'{t}_chance']:.2f})")
    cells += [f"{r['median_x_chance']:.1f}×", f"{r['min_x_chance']:.1f}×", r["verdict"]]
    body.append(cells)

md = [f"**{cap}**\n", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(c) + " |" for c in body]
Path("my_hw1/head_types.md").write_text("\n".join(md) + "\n", encoding="utf-8")
title, _, rest = cap.partition(". ")
html = ["<!doctype html><html><head><meta charset='utf-8'><title>Attention head types</title><style>",
        "body{font-family:Calibri,Arial,sans-serif;margin:24px;background:#fff;color:#000}",
        "table{border-collapse:collapse;margin:6px 0 20px;font-size:10pt}",
        "th,td{border:1px solid #000;padding:3px 6px;text-align:right}th{background:#e8e8e8;text-align:center}",
        "td:nth-child(-n+4){text-align:left}p{font-size:10pt;max-width:1000px;margin:0 0 4px}",
        "img{max-width:1000px;width:100%}</style></head><body>",
        f"<p><b>{escape(title)}.</b> {escape(rest)}</p><table>",
        "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in head) + "</tr>"]
for i, c in enumerate(body):
    # one type spans two rows (two seeds): merge its type/target/verdict cells
    if i % 2 == 0:
        html.append("<tr>" + f"<td rowspan=2>{escape(c[0])}</td><td rowspan=2>{escape(c[1])}</td>"
                    + "".join(f"<td>{escape(x)}</td>" for x in c[2:-1]) + f"<td rowspan=2>{escape(c[-1])}</td></tr>")
    else:
        html.append("<tr>" + "".join(f"<td>{escape(x)}</td>" for x in c[2:-1]) + "</tr>")
html += ["</table>", "<p><b>Figure.</b> Same numbers as Table 9: bars = attention weight on the target, black tick = "
         "chance for that text.</p>", "<img src='head_types.png' alt='attention head types bar chart'>",
         "</body></html>"]
Path("my_hw1/head_types.html").write_text("\n".join(html), encoding="utf-8")

# ---------------- figure ----------------
COLORS = {"shakespeare": "#2a78d6", "prose": "#eb6834", "code": "#1baf7a", "multiline": "#eda100"}  # validated
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
nT, bw, gap = len(TEXTS), 0.19, 0.012
for ax, seed in zip(axes, SEEDS):
    ax.set_facecolor(SURFACE)
    o = out[out.seed == seed].reset_index(drop=True)
    for i, r in o.iterrows():
        for j, t in enumerate(TEXTS):
            w = r[f"{t}_w"]
            x = i + (j - (nT - 1) / 2) * bw
            if np.isnan(w):
                ax.text(x, 0.012, "n/a", ha="center", va="bottom", fontsize=6.5, color=INK2, rotation=90)
                continue
            ax.bar(x, w, width=bw - gap, color=COLORS[t], label=TEXT_LABEL[t] if i == 0 else None, zorder=2)
            ax.plot([x - bw / 2 + 0.02, x + bw / 2 - 0.02], [r[f"{t}_chance"]] * 2, color=INK, lw=1.6, zorder=3,
                    solid_capstyle="butt", label="chance (uniform attention)" if (i == 0 and j == 0) else None)
    ax.set_xticks(range(len(o)))
    ax.set_xticklabels([f"{r['type']}\nhead {r['head']}\n({r['verdict']})" for _, r in o.iterrows()],
                       fontsize=8.5, color=INK)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_title(SEEDS[seed], color=INK, fontsize=10.5, loc="left")
    ax.grid(axis="y", color="#e4e3df", lw=0.8, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.set_ylim(0, 0.75)
axes[0].set_ylabel("average attention weight on the target position")
handles, labels = axes[0].get_legend_handles_labels()
order = [labels.index(TEXT_LABEL[t]) for t in TEXTS] + [labels.index("chance (uniform attention)")]
fig.legend([handles[k] for k in order], [labels[k] for k in order], loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
fig.suptitle("Attention-head types: weight on the target vs. chance, strongest head per type",
             y=1.07, fontsize=11.5, color=INK)
fig.tight_layout()
fig.savefig("my_hw1/head_types.png", dpi=170, bbox_inches="tight", facecolor=SURFACE)
print("\n".join(md))
