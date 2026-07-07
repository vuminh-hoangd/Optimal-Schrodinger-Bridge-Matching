import os, sys
sys.path.append("..")
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.lightsbm_vm_v1 import LightSBM
from notebooks.notebooks_utils import SwissRollSampler, StandardNormalSampler

dim = 2
eps = 0.01
n_potentials = 250
S_init = 1
T = 1
device = "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

sampler_x = StandardNormalSampler(dim=dim, device=device)
sampler_y = SwissRollSampler(device=device)

model = LightSBM(
    dim=dim,
    n_potentials=n_potentials,
    epsilon=eps,
    T=T,
    S_diagonal_init=S_init,
    is_diagonal=True,
)
model.init_mu_by_samples(sampler_y.sample(n_potentials))
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

T_horizon = model.T_horizon.item()
eps_val = model.epsilon.item()
max_iter = 35000
batch_size = 2048
safe_t = 1e-2

best_loss = float("inf")
best_state = None
best_iter = 0
no_improve_count = 0

pbar = tqdm(range(1, max_iter + 1))
for i in pbar:
    x_0 = sampler_x.sample(batch_size).to(device)
    x_1 = sampler_y.sample(batch_size).to(device)
    t = torch.rand(batch_size, 1, device=device) * (T_horizon - safe_t)
    x_t = (
        (t / T_horizon) * x_1
        + ((T_horizon - t) / T_horizon) * x_0
        + torch.sqrt(eps_val * t * (T_horizon - t) / T_horizon) * torch.randn_like(x_0)
    )
    predicted_drift = model.get_drift(x_t, t.squeeze())
    target_drift = (x_1 - x_t) / (T_horizon - t)
    loss = F.mse_loss(target_drift, predicted_drift)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if not torch.isfinite(loss):
        print(f"diverged at {i}"); break

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_iter = i
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if i % 200 == 0:
        pbar.set_description(f"loss {loss.item():.4f} | best {best_loss:.4f} @ {best_iter}")
        if best_iter <= (i - 200):
            no_improve_count += 1
        else:
            no_improve_count = 0
        if no_improve_count >= 10:
            print(f"\nEarly stopping @ iter {i} — best {best_loss:.4f} @ iter {best_iter}")
            break

model.load_state_dict(best_state)
print(f"\nRestored best model from iter {best_iter} with loss {best_loss:.4f}")

# plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6.75), dpi=200)
for ax in axes:
    ax.grid(zorder=-20)

x_samples = sampler_x.sample(2048)
y_samples = sampler_y.sample(2048)
tr_samples = torch.tensor(
    [[0.0, 0.0], [2.05, -1], [-2.05, 1.021], [2, 1.5]], device=device,
)
tr_samples = tr_samples[None].repeat(3, 1, 1).reshape(12, 2)

axes[0].scatter(x_samples[:,0], x_samples[:,1], alpha=0.3, c="g", s=32,
                edgecolors="black", label=r"Input distribution $\pi_0$")
axes[0].scatter(y_samples[:,0], y_samples[:,1], c="orange", s=32,
                edgecolors="black", label=r"Target distribution $\pi_1$")

y_pred = model(x_samples)
axes[1].scatter(y_pred[:,0], y_pred[:,1], c="yellow", s=32,
                edgecolors="black", label="Fitted distribution", zorder=1)

trajectory = model.sample_bridge_trajectory(tr_samples, n_steps=200).detach().cpu()
axes[1].scatter(tr_samples[:,0].cpu(), tr_samples[:,1].cpu(), c="g", s=128,
                edgecolors="black", label=r"Trajectory start ($x \sim \pi_0$)", zorder=3)
axes[1].scatter(trajectory[:,-1,0], trajectory[:,-1,1], c="red", s=64,
                edgecolors="black", label="Trajectory end (fitted)", zorder=3)
for j in range(trajectory.shape[0]):
    axes[1].plot(trajectory[j,:,0], trajectory[j,:,1], "grey", linewidth=0.8, zorder=2,
                 label="Exact bridge trajectory" if j == 0 else None)

for ax in axes:
    ax.set_xlim([-2.5, 2.5]); ax.set_ylim([-2.5, 2.5])
    ax.legend(loc="lower left")
fig.tight_layout(pad=0.1)
plt.title(f"epsilon = {eps}")
plt.savefig("swiss_roll_v1-1.png", dpi=150)
print("saved swiss_roll_v1.png")
