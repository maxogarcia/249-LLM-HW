import json, math
import matplotlib.pyplot as plt

def read_jsonl(path):
    return [json.loads(line) for line in open(path)]

for name in ["lstm_s1337", "lstm_512_s1337"]:
    d = f"outputs/nanolm/sweeps/char_models/{name}"
    ev = read_jsonl(f"{d}/eval_log.jsonl")          # keys: train_tokens, bpb, ...
    tr = read_jsonl(f"{d}/train_log.jsonl")         # keys: step, loss (nats), ...
    line, = plt.plot([e["train_tokens"] / 1e6 for e in ev], [e["bpb"] for e in ev], label=f"{name} val")
    # training loss in bpb (nats / ln 2) against tokens (step * 8192), dashed, same color as val
    plt.plot([t["step"] * 8192 / 1e6 for t in tr], [t["loss"] / math.log(2) for t in tr],
             linestyle="--", color=line.get_color(), alpha=0.7, label=f"{name} train")
plt.xlabel("training tokens (millions)"); plt.ylabel("bits per byte"); plt.legend()
plt.ylim(1.0, 3.0)   # skip the first few very-high training-loss points so the gap is visible
plt.savefig("my_hw1/lstm_overfit.png", dpi=150)
