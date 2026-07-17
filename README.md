# Optimal Schrödinger Bridge Matching

---

## Problem Formulation

Given two marginal distributions $\pi_0, \pi_T \in \mathcal{P}(\mathbb{R}^d)$ and a reference path measure $Q \in \mathcal{P}(C([0,T];\mathbb{R}^d))$ defined by the SDE $dX_t = f(X_t,t)\,dt + \sigma_tdB_t$, the **dynamic Schrödinger Bridge** seeks the path measure $\mathbb{P}^\star$ of minimal KL divergence from $\mathbb{Q}$ subject to the marginal constraints:

$$\mathbb{P}^\star = \arg\min_{\mathbb{P}^u \in \mathcal{P}(C([0,T];\mathbb{R}^d))} \mathrm{KL}(\mathbb{P}^u  \| \mathbb{Q} ) \quad \text{s.t.} \quad \mathbb{P}^u _0 = \pi_0,\; \mathbb{P}^u _T = \pi_T$$

By Girsanov's theorem, the path-space KL divergence reduces to a kinetic energy cost over the control drift $u$:

$$\mathrm{KL}(\mathbb{P}^u \| \mathbb{Q}) = \mathbb{E}_{X_{0:T} \sim \mathbb{P}^u}\left[\int_0^T \frac{1}{2}\|u(X_t, t)\|^2 dt\right]$$

---

## Wiener Prior

When $\mathbb{Q} = \mathbb{W}^\epsilon$ with $f \equiv 0$ and $\sigma_t = \sqrt{\epsilon}$, the problem reduces to learning the **adjusted Schrödinger potential** $v^\star : \mathbb{R}^d \to \mathbb{R}_+$, which fully determines both the optimal coupling and the optimal drift of $\mathbb{P}^*$:

$$
u^*(x, t) = \sqrt{\epsilon} \nabla_x \log \left( \int_{\mathbb{R}^d} \mathcal{N}(x_T \mid x, (T-t)\mathbf{I}_d) e^{\frac{|x_T|^2}{2\epsilon}} v^*(x_T) \, dx_T \right)
$$
---

## LightSB-M

**LightSB-M** parameterizes $v_\theta$ as a Gaussian mixture and optimizes the bridge matching loss:

$$\mathcal{L}(\theta) = \frac{1}{2\epsilon}\int_0^T \mathbb{E}\left[\left\|\sigma_t u_{v_\theta}(X_t,t) - \frac{X_T - X_t}{T - t}\right\|^2\right]dt$$

---

## References

- Gushchin et al. (2024). *Light and Optimal Schrödinger Bridge Matching*. ICML 2024.
- Korotin et al. (2024). *Light Schrödinger Bridge*. ICLR 2024.
- Tang (2026). *Foundations of Schrödinger Bridges for Generative Modeling*.
