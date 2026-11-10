# CMPE 249 — LLM Homework 1: Language models from scratch

**Student:** Maximilian Garcia

This folder holds my work for HW1 of the CMPE 249 LLM module. The assignment is defined in
`../TUTORIAL_nanolm.md` (section *HW 1 — Language models from scratch*). Everything here runs on the
course's `nanolm` library (`../nanolm/`), which comes from the course repository
[lkk688/DeepDataMiningLearning](https://github.com/lkk688/DeepDataMiningLearning). I used it unmodified, at
commit `8418fc1`. My own files are this folder (`my_hw1/`) and my sweep config
`../nanolm/configs/sweeps/my_context.yaml`.

## Setup

| | |
|---|---|
| Data | Tiny Shakespeare, character-level tokenizer (V = 65); validation = last 10% of the text (fixed by the data, not the seed) |
| Budget | 2000 steps × 8192 tokens = **16.4M training tokens per run** (≈16 passes over the ≈1.0M-character training split) |
| Seeds | 1337 and 42 for every configuration; mean ± std and paired t-tests from `nanolm.analyze` |
| Compute | FLOPs ≈ 6 × params × tokens: 0.03–0.42 PFLOP per run |
| Hardware | NVIDIA GeForce RTX 4080 Laptop GPU, bfloat16 autocast, `torch.compile`; PyTorch 2.11.0+cu128, Python 3.11.9, Windows |
| Wall-clock | 28 sweep runs, 37 minutes of training in total |

## What was done

### Part 1: metrics by hand
I recomputed loss, perplexity, bits/token and bits-per-byte from `metrics.json` of
`gpt_modern_s1337` and checked them against the logged values. I also measured the validation loss of
untrained models (`train.max_steps=1`) for the char and GPT-2 BPE tokenizers and compared each with ln V.
Run records: `run_records/my_untrained/`.

### Part 2: RNN vs. LSTM vs. GRU vs. Transformer
Six architectures × 2 seeds from `sweeps/01_char_models.yaml`, all with the same token budget.

| model | params | final val bpb (mean ± std) |
|---|---:|---:|
| Transformer 4×256, RMSNorm + RoPE + SwiGLU | 3.16M | **2.1082 ± 0.0079** |
| LSTM 2×256 | 1.07M | 2.1384 ± 0.0008 |
| GRU 2×256 | 0.81M | 2.1631 ± 0.0041 |
| LSTM 2×512 | 4.24M | 2.1745 ± 0.0067 |
| Transformer 4×256, LayerNorm + learned pos + GELU | 3.44M | 2.1793 ± 0.0043 |
| Vanilla RNN 2×256 | 0.28M | 2.4378 ± 0.0040 |

Findings:
- **The vanilla RNN underfits.** Its train/val gap is only +0.09 bpb, the smallest of all six models.
  Its gradient vanishes through the tanh recurrence, while the gated cells have an additive path.
- **The larger LSTM overfits.** It is the best model early in training, but its validation bpb bottoms out
  at ≈8–10M tokens and then rises, while its training loss keeps falling (final gap +0.69 bpb). It ends
  0.036 bpb worse than the 2×256 LSTM.

Files: `part2_table.py` → `part2_table.md/.csv`; `curves.py` → `lstm_overfit.png`;
`char_models_summary.csv` (from `nanolm.analyze`).

### Part 3: context length
The LSTM and the modern Transformer at `seq_len` = 64, 128, 256 and 512, × 2 seeds, from my sweep
`my_context.yaml`. `batch_size × seq_len` is held at 8192, so every run sees the same tokens.
- **Transformer:** stops improving after T ≈ 128. The later changes are within twice the seed noise floor.
- **LSTM:** bpb keeps falling to T = 512, but a per-position analysis shows that both models' loss is flat
  beyond ≈64 characters of history. The LSTM's 256→512 gain comes mainly from having fewer "cold-start"
  characters per evaluation window: the T = 512 curve averaged over 256 positions predicts 2.1388 bpb, and
  the real T = 256 model scored 2.1387.
- **The Transformer uses its context better:** its loss is lower at every context length and every position.

Files: `plot_context.py` → `context.png`, `context_table.md/.csv`;
`context_doubling.py` → `context_doubling.md/.csv`; `position_loss.py` → `position_loss.png/.md/.csv`;
`my_context_summary.csv`.

### Part 4: attention heads
Attention maps of `gpt_modern_s1337` from `python -m nanolm.viz attention` on three kinds of text:
Shakespeare (`attn_shakespeare.png`), modern prose (`attn_prose.png`) and code (`attn_code.png`).
`*_report.png` are the same maps with the empty band under the title trimmed. Each head is then scored by
its average attention weight on a target position and compared with chance (average of 1/(t+1)), on four
texts and both seeds:

| head type | strongest head (s1337 / s42) | × chance | verdict |
|---|---|---:|---|
| previous character (t−1) | L0H1 / L0H2 | 4.7–7.5× | found |
| two characters back (t−2) | L0H3 / L0H1 | 4.2–5.4× | found |
| line start (most recent newline) | L3H2 / L1H0 | 5.6–10.1× | found |
| word start (4th letter on) | L1H3 / L0H0 | 0.8–4.4× | weak |
| position 0 | — | ≈1× | not found |

Files: `head_stats.py` → `head_stats.md/.csv` (every head × text × measure);
`head_summary.py` → `head_types.md/.csv`, `head_types.png`.

### Extra credit: a GRU, gate by gate
`gru.py` implements a one-layer GRU as a Python loop over time steps in PyTorch's convention (gates in r,
z, n order, with the reset gate also scaling the hidden bias b_hn). It matches `nn.GRU` to ≈1e-16 in
float64. `gru_extra.py` also compares all gradients. In float64 they match to ≈1e-16. On the GPU in
float32 the difference grows to ≈1e-4 because of TF32; turning TF32 off shrinks it about 13–70×.
Results: `gru_extra.md`.

## Run records (`run_records/`)

Copies of all 32 training runs, without the `model.pt` checkpoints. Each run folder has:
`meta.json` (date and time, machine, GPU, library versions, course-repo commit and the exact command that
was run), `config.yaml` (every setting), `train_log.jsonl` / `eval_log.jsonl` (loss during training),
`stdout.log` (console output), `metrics.json` (final results and training time) and `sample.txt`.

| folder | what | dates |
|---|---|---|
| `2026-11-05_23-08-39_smoke`, `2026-11-06_21-41-03_smoke` | setup smoke tests | Nov 5–6, 2026 |
| `my_untrained/` | Part 1 untrained-model losses | Nov 6, 2026 |
| `sweeps/char_models/` | Part 2, 12 runs | Nov 5, 2026 |
| `sweeps/my_context/` | Part 3, 16 runs | Nov 6, 2026 |

## Reproducing

From the `llm/` folder, with the course environment installed:

```powershell
python -m nanolm.sweep --sweep nanolm/configs/sweeps/01_char_models.yaml
python -m nanolm.sweep --sweep nanolm/configs/sweeps/my_context.yaml
python -m nanolm.analyze outputs/nanolm/sweeps/char_models --baseline gpt_textbook --metric val_bpb
python -m nanolm.analyze outputs/nanolm/sweeps/my_context --baseline gpt_modern_T256 --metric val_bpb

python my_hw1/part2_table.py
python my_hw1/curves.py
python my_hw1/plot_context.py
python my_hw1/context_doubling.py
python my_hw1/position_loss.py       # needs the model.pt checkpoints from the runs above
python my_hw1/head_stats.py          # needs the model.pt checkpoints
python my_hw1/head_summary.py
python my_hw1/gru.py
python my_hw1/gru_extra.py
```

The analysis scripts read from `outputs/nanolm/...`, which the sweeps above recreate. The checkpoints
are not committed (the assignment says not to submit them).
