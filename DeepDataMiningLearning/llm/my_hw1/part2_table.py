"""Part 2 results table: RNN vs GRU vs LSTM vs Transformer (char-level Tiny Shakespeare).

Run from the llm folder:  python my_hw1/part2_table.py
Writes my_hw1/part2_table.csv (full precision) and my_hw1/part2_table.md (report-ready).
"""
import json, math, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, ".")                       # so we can reuse the course's own paired t-test
from nanolm.analyze import load_runs, paired_test

SWEEP = "outputs/nanolm/sweeps/char_models"
BASELINE = "gpt_textbook"
LN2 = math.log(2)
TRAIN_TAIL = 5                                # average the last 5 train-log entries (= last 250 steps)

DESCRIPTION = {
    "gpt_modern":   "Transformer 4×256, RMSNorm + RoPE + SwiGLU",
    "lstm":         "LSTM 2×256",
    "gru":          "GRU 2×256",
    "lstm_512":     "LSTM 2×512",
    "gpt_textbook": "Transformer 4×256, LayerNorm + learned pos + GELU",
    "rnn":          "Vanilla RNN 2×256",
}

def read_jsonl(path):
    return [json.loads(line) for line in open(path)]

# ---- per run (one row per model × seed) ----
df = load_runs([SWEEP])
rows = []
for _, r in df.iterrows():
    ev = read_jsonl(Path(r["run_dir"]) / "eval_log.jsonl")
    tr = read_jsonl(Path(r["run_dir"]) / "train_log.jsonl")
    best = min(ev, key=lambda e: e["bpb"])
    train_bpb = sum(t["loss"] for t in tr[-TRAIN_TAIL:]) / TRAIN_TAIL / LN2
    rows.append({
        "name": r["name"], "seed": r["seed"],
        "params": r["params"], "params_non_emb": r["params_non_embedding"],
        "tokens": r["tokens_seen"], "train_min": r["train_time_sec"] / 60,
        "val_bpb": r["val_bpb"],                       # final, full validation set
        "best_val_bpb": best["bpb"], "best_at_Mtok": best["train_tokens"] / 1e6,
        "train_bpb": train_bpb,
        "gap": r["val_bpb"] - train_bpb,                # val - train: big & growing = overfitting
    })
runs = pd.DataFrame(rows)

# ---- seed noise floor (same definition as nanolm.analyze) ----
noise = runs.groupby("name")["val_bpb"].std(ddof=1).median()
threshold = 2 * noise

# ---- aggregate over seeds ----
g = runs.groupby("name")
t = pd.DataFrame({
    "model": [DESCRIPTION.get(n, n) for n in g.size().index],
    "params_M": g["params"].first() / 1e6,
    "params_non_emb_M": g["params_non_emb"].first() / 1e6,
    "train_PFLOP": 6 * g["params"].first() * g["tokens"].first() / 1e15,   # FLOPs ≈ 6·N·D
    "val_bpb_mean": g["val_bpb"].mean(),
    "val_bpb_std": g["val_bpb"].std(ddof=1),
    "best_val_bpb_mean": g["best_val_bpb"].mean(),
    "best_at_Mtok": g["best_at_Mtok"].mean(),
    "train_bpb_mean": g["train_bpb"].mean(),
    "gap_mean": g["gap"].mean(),
    "train_min": g["train_min"].mean(),
}, index=g.size().index)

deltas, pvals = [], []
for name in t.index:
    if name == BASELINE:
        deltas.append(0.0); pvals.append(float("nan")); continue
    d, p, _ = paired_test(runs, "val_bpb", BASELINE, name)
    deltas.append(d); pvals.append(p)
t[f"delta_vs_{BASELINE}"] = deltas
t["paired_p"] = pvals
t["resolved"] = [("—" if n == BASELINE else ("yes" if abs(d) >= threshold else "NOT resolved"))
                 for n, d in zip(t.index, deltas)]
t = t.sort_values("val_bpb_mean")
t.to_csv("my_hw1/part2_table.csv")

# ---- key pairwise comparisons for the report questions ----
pairs = [("lstm", "lstm_512", "Is the larger LSTM better?"),
         ("lstm", "gru",      "GRU vs LSTM (same width)"),
         ("lstm", "rnn",      "Do gates matter?"),
         ("gpt_textbook", "gpt_modern", "Does the modern recipe matter?"),
         ("lstm", "gpt_modern", "Best RNN vs best Transformer")]

# ---- build the three tables as (caption, header, rows), then render as Markdown and HTML ----
def f(x, nd=4): return "—" if pd.isna(x) else f"{x:.{nd}f}"
tables = []

rows1 = []
for n, r in t.iterrows():
    delta = r[f"delta_vs_{BASELINE}"]
    rows1.append([r["model"], f"{r['params_M']:.2f}", f"{r['train_PFLOP']:.2f}",
                  f"{r['val_bpb_mean']:.4f} ± {r['val_bpb_std']:.4f}",
                  "0 (baseline)" if n == BASELINE else f"{delta:+.4f}", f(r["paired_p"], 3), r["resolved"],
                  f"{r['best_val_bpb_mean']:.4f}", f"{r['best_at_Mtok']:.1f}",
                  f"{r['train_bpb_mean']:.3f}", f"{r['gap_mean']:+.3f}", f"{r['train_min']:.2f}"])
tables.append((
    "Table 1. Character-level Tiny Shakespeare, 16.4M training tokens per run, 2 seeds (1337, 42). "
    "Val bpb is the final value on the full validation set (mean ± std over seeds). "
    f"Δ and paired p are vs. the GPT-2-style Transformer. Seed noise floor = {noise:.4f} bpb; differences "
    f"smaller than 2× that ({threshold:.4f}) are marked not resolved. Train bpb = mean of the last "
    f"{TRAIN_TAIL} logged training losses (last 250 steps) ÷ ln 2. Gap = final val − train. "
    "FLOPs ≈ 6 × params × tokens.",
    ["model", "params (M)", "train FLOPs (PFLOP)", "final val bpb", "Δ vs GPT-2-style", "paired p",
     "resolved?", "best val bpb", "best at (M tokens)", "train bpb", "val − train gap", "train time (min)"],
    rows1))

rows2 = []
for a, b, q in pairs:
    d, p, _ = paired_test(runs, "val_bpb", a, b)
    rows2.append([q, DESCRIPTION[a], DESCRIPTION[b], f"{d:+.4f}", f(p, 3),
                  "yes" if abs(d) >= threshold else "NOT resolved"])
tables.append(("Table 2. Pairwise comparisons for the report questions (paired over the same seeds).",
               ["question", "A", "B", "Δ (B − A) bpb", "paired p", "resolved?"], rows2))

rows3 = [[r["name"], str(r["seed"]), f"{r['val_bpb']:.4f}", f"{r['best_val_bpb']:.4f}",
          f"{r['best_at_Mtok']:.1f}", f"{r['train_bpb']:.3f}", f"{r['gap']:+.3f}"]
         for _, r in runs.sort_values(["name", "seed"]).iterrows()]
tables.append(("Table 3. Per-seed detail.",
               ["run", "seed", "final val bpb", "best val bpb", "best at (M tokens)", "train bpb", "gap"], rows3))

# Markdown
L = []
for cap, head, rows in tables:
    L.append(f"**{cap}**\n")
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    L += ["| " + " | ".join(r) + " |" for r in rows]
    L.append("")
Path("my_hw1/part2_table.md").write_text("\n".join(L), encoding="utf-8")

# HTML: open in a browser, select a table, copy, and paste into Word / Google Docs
from html import escape
H = ["<!doctype html><html><head><meta charset='utf-8'><title>Part 2 tables</title><style>",
     "body{font-family:Calibri,Arial,sans-serif;margin:24px;background:#fff;color:#000}",
     "table{border-collapse:collapse;margin:6px 0 28px;font-size:10pt}",
     "th,td{border:1px solid #000;padding:3px 6px;text-align:right}",
     "th{background:#e8e8e8;text-align:center}td:first-child{text-align:left}",
     "p.cap{font-size:10pt;max-width:900px;margin:0}</style></head><body>"]
for cap, head, rows in tables:
    title, _, rest = cap.partition(". ")
    H.append(f"<p class='cap'><b>{escape(title)}.</b> {escape(rest)}</p><table>")
    H.append("<tr>" + "".join(f"<th>{escape(h)}</th>" for h in head) + "</tr>")
    H += ["<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in rows]
    H.append("</table>")
H.append("</body></html>")
Path("my_hw1/part2_table.html").write_text("\n".join(H), encoding="utf-8")
print("wrote my_hw1/part2_table.md, part2_table.html, part2_table.csv")
