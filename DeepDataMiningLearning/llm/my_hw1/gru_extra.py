"""Extra credit, "go further": (1) gradients of GRUScratch vs nn.GRU in float64,
(2) forward + gradient differences on the GPU in float32, with and without TF32.

Run from the llm folder:  python my_hw1/gru_extra.py
Writes my_hw1/gru_extra.md
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, "my_hw1")
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):          # gru.py prints its own check on import
    from gru import GRUScratch

D, H, B, T = 8, 16, 4, 10
PAIRS = [("weight_ih_l0", "W_x.weight"), ("bias_ih_l0", "W_x.bias"),
         ("weight_hh_l0", "W_h.weight"), ("bias_hh_l0", "W_h.bias")]

def make(dtype, device):
    torch.manual_seed(0)
    ref = nn.GRU(D, H, batch_first=True).to(device=device, dtype=dtype)
    mine = GRUScratch(D, H).to(device=device, dtype=dtype)
    with torch.no_grad():
        for r, m in PAIRS:
            mine.get_parameter(m).copy_(ref.get_parameter(r))
    return ref, mine

def compare(dtype, device):
    """Same input, same loss through both models; return max |diff| of output, h_T, x.grad, every weight grad."""
    ref, mine = make(dtype, device)
    torch.manual_seed(1)
    x0 = torch.randn(B, T, D, dtype=dtype, device=device)
    target = torch.randn(B, T, H, dtype=dtype, device=device)
    res = {}
    outs = {}
    for name, model in [("ref", ref), ("mine", mine)]:
        x = x0.clone().requires_grad_(True)
        out, h = model(x)
        loss = ((out - target) ** 2).mean() + h.pow(2).sum()      # depends on every step and on h_T
        loss.backward()
        outs[name] = (out.detach(), h.detach().reshape(B, H), x.grad)
    res["output"] = (outs["ref"][0] - outs["mine"][0]).abs().max().item()
    res["h_T"] = (outs["ref"][1] - outs["mine"][1]).abs().max().item()
    res["x.grad"] = (outs["ref"][2] - outs["mine"][2]).abs().max().item()
    for r, m in PAIRS:
        res[f"grad {r}"] = (ref.get_parameter(r).grad - mine.get_parameter(m).grad).abs().max().item()
    return res

rows = {"CPU float64": compare(torch.float64, "cpu")}
gpu = torch.cuda.is_available()
if gpu:
    dev = torch.cuda.get_device_name(0)
    for tf32 in [True, False]:
        torch.backends.cudnn.allow_tf32 = tf32           # used by nn.GRU's cuDNN kernel
        torch.backends.cuda.matmul.allow_tf32 = tf32     # used by GRUScratch's nn.Linear matmuls
        rows[f"GPU float32, TF32 {'on' if tf32 else 'off'}"] = compare(torch.float32, "cuda")
    torch.backends.cudnn.allow_tf32 = True               # restore PyTorch defaults
    torch.backends.cuda.matmul.allow_tf32 = False
    rows["CPU float32 (reference point)"] = compare(torch.float32, "cpu")

cols = list(next(iter(rows.values())).keys())
L = [f"**Table 10. GRUScratch vs nn.GRU: max |difference| (D={D}, H={H}, batch {B}, {T} steps). "
     f"Loss = MSE(output, random target) + ‖h_T‖², back-propagated through both models."
     + (f" GPU: {dev}." if gpu else " No GPU available.") + "**\n",
     "| setting | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
for k, r in rows.items():
    L.append(f"| {k} | " + " | ".join(f"{v:.1e}" for v in r.values()) + " |")
Path("my_hw1/gru_extra.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
