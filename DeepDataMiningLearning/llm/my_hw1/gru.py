import torch
import torch.nn as nn

class GRUScratch(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.W_x = nn.Linear(in_dim, 3 * hidden)   # input part of all 3 gates
        self.W_h = nn.Linear(hidden, 3 * hidden)   # hidden part of all 3 gates

    def forward(self, x):                          # x: [B, T, D]
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden)
        outs = []
        for t in range(T):
            xr, xz, xn = self.W_x(x[:, t]).chunk(3, dim=-1)   # W_i* x + b_i*
            hr, hz, hn = self.W_h(h).chunk(3, dim=-1)         # W_h* h + b_h*
            r = torch.sigmoid(xr + hr)             # reset gate
            z = torch.sigmoid(xz + hz)             # update gate
            n = torch.tanh(xn + r * hn)            # candidate; r also scales b_hn (it's inside hn)
            h = (1 - z) * n + z * h                # z weights the OLD state
            outs.append(h)
        return torch.stack(outs, dim=1), h

# --- verification ---
torch.manual_seed(0)
D, H = 8, 16
ref = nn.GRU(D, H, batch_first=True).double()
mine = GRUScratch(D, H).double()
with torch.no_grad():
    mine.W_x.weight.copy_(ref.weight_ih_l0)
    mine.W_x.bias.copy_(ref.bias_ih_l0)
    mine.W_h.weight.copy_(ref.weight_hh_l0)
    mine.W_h.bias.copy_(ref.bias_hh_l0)
x = torch.randn(4, 10, D, dtype=torch.double)
out_ref, h_ref = ref(x)
out_mine, h_mine = mine(x)
print("max |diff| output:", (out_ref - out_mine).abs().max().item())
print("allclose:", torch.allclose(out_ref, out_mine, atol=1e-10))
