[README.md](https://github.com/user-attachments/files/29735767/README.md)
# Optimal-Schr-dinger-Bridge-Matching
corrected math derivation of LightSB-M
# Schrödinger Bridge Matching

**Author:** Hoang Dung Vu Minh

---

## Problem Formulation

Given two marginal distributions $\pi_0, \pi_T \in \mathcal{P}(\mathbb{R}^d)$ and a reference path measure $Q \in \mathcal{P}(\mathcal{C}([0,T];\mathbb{R}^d))$ defined by the SDE $dX_t = f(X_t,t)\,dt + \sigma_t\,dB_t$, the **dynamic Schrödinger Bridge** seeks the path measure $P^\star$ of minimal KL divergence from $Q$ subject to the marginal constraints:

$$\mathbb{P}^\star = \arg\min_{\mathbb{P}^u \in \mathcal{P}(\mathcal{C}([0,T];\mathbb{R}^d))} \mathrm{KL}(P^u \| Q) \quad \text{s.t.} \quad P^u_0 = \pi_0,\; P^u_T = \pi_T$$

By Girsanov's theorem, the path-space KL divergence reduces to a kinetic energy cost over the control drift $u$:

$$\mathrm{KL}(\mathbb{P}^u \| \mathbb{Q}) = \mathbb{E}_{X_{0:T} \sim \mathbb{P}^u}\!\left[\int_0^T \frac{1}{2}\|u(X_t, t)\|^2\, dt\right]$$

---

## Wiener Prior

When $\mathbb{Q} = \mathbb{W}^\epsilon$ with $f \equiv 0$ and $\sigma_t = \sqrt{\epsilon}$, the problem reduces to learning the **adjusted Schrödinger potential** $v^\star : \mathbb{R}^d \to \mathbb{R}_+$, which fully determines both the optimal coupling and the optimal drift.

---

## LightSB-M

**LightSB-M** parameterizes $v_\theta$ as a Gaussian mixture and optimizes the bridge matching loss:

$$\mathcal{L}(\theta) = \frac{1}{2\epsilon}\int_0^T \mathbb{E}\!\left[\left\|\sigma_t u_{v_\theta}(X_t,t) - \frac{X_T - X_t}{T - t}\right\|^2\right]dt$$

Both the drift and the conditional sampler $(\pi^{SB}_{v_\theta})_{T|0}(\cdot \mid x_0)$ are available in closed form, enabling exact trajectory generation without an SDE solver.

---

## References

- Gushchin et al. (2024). *Light and Optimal Schrödinger Bridge Matching*. ICML 2024.
- Korotin et al. (2024). *Light Schrödinger Bridge*. ICLR 2024.
- Tang (2026). *Foundations of Schrödinger Bridges for Generative Modeling*.
