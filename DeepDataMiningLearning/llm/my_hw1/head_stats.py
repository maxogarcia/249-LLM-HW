"""Part 4b: support each attention-head type with a number (average attention weight on the relevant
position, as in tutorial §1.6.2), with the chance level for comparison.

Head types measured, for every (layer, head):
  prev        weight on position t-1                                     queries t = 1..T-1
  word_start  weight on the first character s of the current word        queries t >= s+2 inside a word
              (the 2nd letter is excluded: there the word start IS t-1, so a previous-char head would score)
  line_start  weight on the most recent "\n" before t                    queries t >= nl+2 (same exclusion)
  first       weight on position 0 (the "nothing better to do" sink)     queries t = 1..T-1
Chance = what a head would score spreading attention evenly over the t+1 allowed positions: mean of 1/(t+1)
over the same queries.

Run from the llm folder:  python my_hw1/head_stats.py
Writes my_hw1/head_stats.csv (every head), head_stats.md and head_stats.html (report tables).
"""
import sys
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from nanolm.generate import load_run
from nanolm.viz import attention_maps

RUNS = {"s1337": "outputs/nanolm/sweeps/char_models/gpt_modern_s1337",
        "s42":   "outputs/nanolm/sweeps/char_models/gpt_modern_s42"}
TEXTS = {   # the three kinds of text used for the maps, plus one multi-line text for the line-start measure
    "shakespeare": "ROMEO:\nBut soft! what light through",
    "prose":       "The bus was late again, so I walked to work.",
    "code":        "def add(a, b):\n    return a + b",
    "multiline":   "ROMEO:\nBut soft!\nJULIET:\nAy me!\nROMEO:\nShe speaks.\nO, speak again!",
}

def is_word(c): return c.isalnum() or c in "_'"

def queries(text):
    """For each metric: list of (query t, key position it should look at)."""
    T = len(text)
    q = {"prev": [(t, t - 1) for t in range(1, T)],
         "prev2": [(t, t - 2) for t in range(2, T)],          # control: a plain "two back" head
         "first": [(t, 0) for t in range(1, T)],
         "word_start": [], "word_start_far": [], "line_start": []}
    s = None
    for t, c in enumerate(text):
        if is_word(c):
            if s is None: s = t
            if t >= s + 2: q["word_start"].append((t, s))
            if t >= s + 3: q["word_start_far"].append((t, s))   # start is >= 3 back: not t-1 or t-2
        else:
            s = None
    nl = None
    for t, c in enumerate(text):
        if nl is not None and t >= nl + 2: q["line_start"].append((t, nl))
        if c == "\n": nl = t
    return q

rows = []
for seed, run in RUNS.items():
    cfg, model, tok, dinfo, _ = load_run(run, device="cpu")
    model.float()
    for tname, text in TEXTS.items():
        with torch.no_grad():
            maps = attention_maps(model, torch.tensor([tok.encode(text)]))
        for metric, qs in queries(text).items():
            if not qs: continue
            chance = np.mean([1 / (t + 1) for t, _ in qs])
            for L, a in enumerate(maps):
                for H in range(a.shape[0]):
                    w = np.mean([a[H, t, k] for t, k in qs])
                    rows.append({"model": seed, "text": tname, "metric": metric, "layer": L, "head": H,
                                 "weight": w, "chance": chance, "x_chance": w / chance, "n_queries": len(qs)})
df = pd.DataFrame(rows)
df.to_csv("my_hw1/head_stats.csv", index=False)

# ---------- report tables ----------
tables = []
LABEL = {"prev": "previous character", "word_start": "first letter of current word",
         "word_start_far": "first letter of current word (4th letter on)",
         "line_start": "most recent newline", "first": "position 0", "prev2": "two characters back"}

# Table A: the best head for each type, on each text, for each model (robustness: same head? same type?)
body = []
for metric, texts in [("prev", ["shakespeare", "prose", "code"]),
                      ("word_start", ["shakespeare", "prose", "code"]),
                      ("word_start_far", ["shakespeare", "prose", "multiline"]),
                      ("line_start", ["multiline", "shakespeare", "code"])]:
    for tname in texts:
        cells = [LABEL[metric], tname]
        sub = df[(df.metric == metric) & (df.text == tname)]
        if sub.empty: continue
        for seed in RUNS:
            r = sub[sub.model == seed].sort_values("weight", ascending=False).iloc[0]
            cells.append(f"L{r['layer']}H{r['head']}: {r['weight']:.2f}")
        r0 = sub.iloc[0]
        cells += [f"{r0.chance:.3f}", str(int(r0.n_queries))]
        body.append(cells)
tables.append(("Table 6. Strongest head of each type (average attention weight on the relevant position) for "
               "each text, in the two seeds of the modern Transformer. Chance = average of 1/(t+1) over the same "
               "queries (uniform attention over the allowed positions). Word-start and line-start exclude the "
               "query right after the target, where it is also t−1.",
               ["head type: attends to", "text", "gpt_modern_s1337", "gpt_modern_s42", "chance", "# queries"], body))

# Table B: the named heads of seed 1337, measured on every text (does the same head keep its job?)
named = []
for metric in ["prev", "word_start", "word_start_far", "line_start"]:
    ref = "multiline" if metric == "line_start" else "shakespeare"
    r = df[(df.model == "s1337") & (df.metric == metric) & (df.text == ref)].sort_values("weight").iloc[-1]
    named.append((metric, int(r["layer"]), int(r["head"])))
body = []
for metric, L, H in named:
    cells = [f"L{L}H{H}", LABEL[metric]]
    for tname in ["shakespeare", "prose", "code", "multiline"]:
        r = df[(df.model == "s1337") & (df.metric == metric) & (df.text == tname) & (df.layer == L) & (df["head"] == H)]
        cells.append("n/a" if r.empty else f"{r.iloc[0].weight:.2f} (chance {r.iloc[0].chance:.2f})")
    body.append(cells)
tables.append(("Table 7. The strongest head of each type in gpt_modern_s1337, measured on every text. "
               "n/a = the text has no such positions (e.g. no newline in the prose sentence).",
               ["head", "attends to", "shakespeare", "prose", "code", "multiline"], body))

# Table C: every head on the Shakespeare text, seed 1337 (full picture, for the appendix)
sub = df[(df.model == "s1337")]
body = []
for L in range(4):
    for H in range(4):
        def g(metric, tname):
            r = sub[(sub.metric == metric) & (sub.text == tname) & (sub.layer == L) & (sub["head"] == H)]
            return f"{r.iloc[0].weight:.2f}"
        body.append([f"L{L}H{H}", g("prev", "shakespeare"), g("prev2", "shakespeare"),
                     g("word_start", "shakespeare"), g("word_start_far", "multiline"),
                     g("line_start", "multiline"), g("first", "shakespeare")])
ch = lambda m, t: f"{sub[(sub.metric == m) & (sub.text == t)].iloc[0].chance:.3f}"
body.append(["chance", ch("prev", "shakespeare"), ch("prev2", "shakespeare"), ch("word_start", "shakespeare"),
             ch("word_start_far", "multiline"), ch("line_start", "multiline"), ch("first", "shakespeare")])
tables.append(("Table 8 (appendix). All 16 heads of gpt_modern_s1337: average attention weight on each target. "
               "prev, two-back, word-start and position-0 measured on the Shakespeare text; word-start (4th letter "
               "on) and line-start on the multi-line text. Two-back is a control: a word-start head should score "
               "high on word start but not on two-back.",
               ["head", "prev char (t−1)", "two back (t−2)", "word start", "word start, 4th letter on",
                "line start (newline)", "position 0"], body))

md, html = [], ["<!doctype html><html><head><meta charset='utf-8'><title>Attention head stats</title><style>",
                "body{font-family:Calibri,Arial,sans-serif;margin:24px;background:#fff;color:#000}",
                "table{border-collapse:collapse;margin:6px 0 28px;font-size:10pt}",
                "th,td{border:1px solid #000;padding:3px 6px;text-align:right}th{background:#e8e8e8;text-align:center}",
                "td:nth-child(-n+2){text-align:left}p{font-size:10pt;max-width:900px;margin:0}</style></head><body>"]
for cap, head, body in tables:
    md += [f"**{cap}**\n", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in body] + [""]
    title, _, rest = cap.partition(". ")
    html.append(f"<p><b>{escape(title)}.</b> {escape(rest)}</p><table>")
    html.append("<tr>" + "".join(f"<th>{escape(h)}</th>" for h in head) + "</tr>")
    html += ["<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in body]
    html.append("</table>")
html.append("</body></html>")
Path("my_hw1/head_stats.md").write_text("\n".join(md), encoding="utf-8")
Path("my_hw1/head_stats.html").write_text("\n".join(html), encoding="utf-8")
print("\n".join(md))
