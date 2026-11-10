import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/nanolm/sweeps/my_context/summary.csv")   # ONE ROW PER RUN (each seed separately)
print(df[["name", "seed", "val_bpb"]])
# names look like "lstm_T64" -> split into architecture and T
df["arch"] = df["name"].str.rsplit("_T", n=1).str[0]
df["T"] = df["name"].str.rsplit("_T", n=1).str[1].astype(int)

# average over the two seeds: one row per (arch, T) with mean and std of val_bpb
agg = df.groupby(["arch", "T"])["val_bpb"].agg(["mean", "std"]).reset_index()
print(agg)

# save the table: wide format, one row per T, mean ± std per architecture
table = agg.pivot(index="T", columns="arch", values=["mean", "std"])
table.to_csv("my_hw1/context_table.csv")
lines = ["| T | " + " | ".join(sorted(agg["arch"].unique())) + " |",
         "|---|" + "---|" * agg["arch"].nunique()]
for T in sorted(agg["T"].unique()):
    cells = [f'{table.loc[T, ("mean", a)]:.4f} ± {table.loc[T, ("std", a)]:.4f}'
             for a in sorted(agg["arch"].unique())]
    lines.append(f"| {T} | " + " | ".join(cells) + " |")
open("my_hw1/context_table.md", "w", encoding="utf-8").write(
    "Validation bpb (mean ± std over 2 seeds)\n\n" + "\n".join(lines) + "\n")

for arch, g in agg.groupby("arch"):
    g = g.sort_values("T")
    plt.errorbar(g["T"], g["mean"], yerr=g["std"], marker="o", capsize=4, label=arch)
plt.xscale("log", base=2)
plt.xlabel("context length seq_len (characters)"); plt.ylabel("validation bpb"); plt.legend()
plt.savefig("my_hw1/context.png", dpi=150)
