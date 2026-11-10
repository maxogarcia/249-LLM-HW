"""Part 3: change in val bpb for each doubling of context, vs. the seed noise floor.

Run from the llm folder:  python my_hw1/context_doubling.py
Writes my_hw1/context_doubling.md / .csv / .html.
"""
import sys
from html import escape
from pathlib import Path
import pandas as pd

sys.path.insert(0, ".")
from nanolm.analyze import load_runs, paired_test

df = load_runs(["outputs/nanolm/sweeps/my_context"])
noise = df.groupby("name")["val_bpb"].std(ddof=1).median()     # same definition as nanolm.analyze
thr = 2 * noise
LABEL = {"lstm": "LSTM 2×256", "gpt_modern": "Transformer (modern) 4×256"}

rows = []
for arch in ["gpt_modern", "lstm"]:
    for a, b in [(64, 128), (128, 256), (256, 512)]:
        na, nb = f"{arch}_T{a}", f"{arch}_T{b}"
        d, p, _ = paired_test(df, "val_bpb", na, nb)          # d = bpb(T=b) - bpb(T=a), paired by seed
        rows.append({"model": LABEL[arch], "doubling": f"{a} → {b}",
                     "bpb_from": df[df.name == na].val_bpb.mean(), "bpb_to": df[df.name == nb].val_bpb.mean(),
                     "delta": d, "paired_p": p, "x_noise": abs(d) / noise,
                     "resolved": "yes" if (d < 0 and abs(d) >= thr) else "NOT resolved"})
t = pd.DataFrame(rows)
t.to_csv("my_hw1/context_doubling.csv", index=False)

cap = (f"Table 4. Change in validation bpb per doubling of context length (negative = better), paired over seeds "
       f"1337 and 42. Seed noise floor = {noise:.4f} bpb (median within-config std); a gain smaller than "
       f"2× that ({thr:.4f}) is marked not resolved.")
head = ["model", "doubling", "bpb before", "bpb after", "Δ bpb", "paired p", "|Δ| / noise", "resolved?"]
body = [[r.model, r.doubling, f"{r.bpb_from:.4f}", f"{r.bpb_to:.4f}", f"{r.delta:+.4f}",
         f"{r.paired_p:.3f}", f"{r.x_noise:.1f}×", r.resolved] for r in t.itertuples()]

md = [f"**{cap}**\n", "| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 2)]
md += ["| " + " | ".join(r) + " |" for r in body]
Path("my_hw1/context_doubling.md").write_text("\n".join(md) + "\n", encoding="utf-8")

title, _, rest = cap.partition(". ")
html = ["<!doctype html><html><head><meta charset='utf-8'><title>Context doubling</title><style>",
        "body{font-family:Calibri,Arial,sans-serif;margin:24px;background:#fff;color:#000}",
        "table{border-collapse:collapse;font-size:10pt}th,td{border:1px solid #000;padding:3px 6px;text-align:right}",
        "th{background:#e8e8e8;text-align:center}td:nth-child(-n+2){text-align:left}p{font-size:10pt;max-width:900px}",
        "</style></head><body>", f"<p><b>{escape(title)}.</b> {escape(rest)}</p><table>",
        "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in head) + "</tr>"]
html += ["<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in body]
html.append("</table></body></html>")
Path("my_hw1/context_doubling.html").write_text("\n".join(html), encoding="utf-8")
print("\n".join(md))
