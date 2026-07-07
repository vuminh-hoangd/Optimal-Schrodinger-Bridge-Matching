"""
Light Schrödinger Bridge Matching (LightSB-M)

Implementation based on:
    - Proposition 1: Closed-form drift under Gaussian mixture potential
    - Proposition 2: Closed-form conditional sampling under GMM parameterization
    - Algorithm 1: LightSB-M training with bridge matching loss

Parameterization:
    The adjusted Schrödinger potential is a Gaussian mixture:
        v_θ(x_T) = Σ_k α_k N(x_T | μ_k, ε Σ_k)

    where Σ_k is the "base" covariance (learnable parameter),
    and ε Σ_k is the full covariance appearing in the density.

Supports:
    - General time horizon T (not restricted to T=1)
    - Diagonal or general SPD covariance Σ_k
    - Two inference modes: Euler-Maruyama SDE / exact bridge sampling
"""

import torch
import torch.nn as nn
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
import math

try:
    import geotorch
    HAS_GEOTORCH = True
except ImportError:
    HAS_GEOTORCH = False


class LightSBM(nn.Module):
    """
    Light Schrödinger Bridge Matching solver with Gaussian mixture potential.

    Parameters
    ----------
    dim : int
        Dimension d of the state space R^d.
    n_potentials : int
        Number of Gaussian mixture components K.
    epsilon : float
        Diffusion coefficient ε > 0.
    T : float
        Time horizon T > 0  (default 1.0).
    is_diagonal : bool
        If True, each Σ_k is diagonal.  If False, Σ_k = R_k diag(e^s_k) R_k^T
        with R_k constrained to be orthogonal (requires geotorch).
    S_diagonal_init : float
        Initial value for the diagonal entries of Σ_k.
    """

    def __init__(
        self,
        dim: int = 2,
        n_potentials: int = 5,
        epsilon: float = 1.0,
        T: float = 1.0,
        is_diagonal: bool = True,
        S_diagonal_init: float = 0.1,
    ):
        super().__init__()
        self.is_diagonal = is_diagonal
        self.dim = dim
        self.n_potentials = n_potentials

        # ---- buffers (not learned) ----
        self.register_buffer("epsilon", torch.tensor(float(epsilon)))
        self.register_buffer("T_horizon", torch.tensor(float(T)))

        # ---- learnable parameters ----
        # Mixture log-weights  log α_k   (K,)
        self.log_alpha = nn.Parameter(
            torch.log(torch.ones(n_potentials) / n_potentials)
        )

        # Mixture centres  μ_k   (K, D)
        self.mu = nn.Parameter(torch.randn(n_potentials, dim))

        # Base covariance stored as log-diagonal  s_k   (K, D)
        # so that σ_{k,i}^2 = exp(s_{k,i}) > 0 always
        self.S_log_diag = nn.Parameter(
            torch.log(S_diagonal_init * torch.ones(n_potentials, dim))
        )

        # Rotation matrices for the non-diagonal case
        if not is_diagonal:
            self.S_rotation = nn.Parameter(
                torch.randn(n_potentials, dim, dim)
            )
            if not HAS_GEOTORCH:
                raise ImportError(
                    "geotorch is required for non-diagonal covariance. "
                    "Install with: pip install geotorch"
                )
            geotorch.orthogonal(self, "S_rotation")

    # ================================================================
    # Initialisation helpers
    # ================================================================

    def init_mu_by_samples(self, samples: torch.Tensor):
        """Initialise μ_k centres from samples of the target distribution."""
        assert samples.shape[0] == self.n_potentials
        self.mu.data = samples.clone().to(self.mu.device)

    init_r_by_samples = init_mu_by_samples

    def set_epsilon(self, new_epsilon: float):
        """Change ε after construction."""
        self.epsilon = torch.tensor(float(new_epsilon), device=self.epsilon.device)

    # ================================================================
    # Parameter access
    # ================================================================

    def get_Sigma_diag(self) -> torch.Tensor:
        """Diagonal entries exp(s_{k,i}) of the base Σ_k.  Shape (K, D)."""
        return torch.exp(self.S_log_diag)

    def get_Sigma(self) -> torch.Tensor:
        """
        Full base covariance Σ_k.
            Diagonal case  → (K, D)      (diagonal entries only)
            General  case  → (K, D, D)   (full SPD matrices)
        """
        S_diag = self.get_Sigma_diag()
        if self.is_diagonal:
            return S_diag
        else:
            R = self.S_rotation  # (K, D, D)
            return (R * S_diag[:, None, :]) @ R.transpose(-1, -2)

    def get_mu(self) -> torch.Tensor:
        """Mixture centres μ_k.  Shape (K, D)."""
        return self.mu

    # ================================================================
    # Forward:  exact conditional sampling  x_T ~ π(x_T | x_0)
    # ================================================================

    @torch.no_grad()
    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Sample  x_T ~ π_{v_θ}^{SB}(· | x_0)  exactly. Stabilized for small epsilon.

        Parameters
        ----------
        x_0 : (N, D)

        Returns
        -------
        x_T : (N, D)
        """
        eps = self.epsilon
        T = self.T_horizon
        mu = self.get_mu()            # (K, D)
        log_alpha = self.log_alpha    # (K,)

        if self.is_diagonal:
            S_diag = self.get_Sigma_diag()  # (K, D)

            # μ_k^T x_0   →  (N, K)
            mu_x = (mu[None, :, :] * x_0[:, None, :]).sum(dim=-1)

            # x_0^T Σ_k x_0  →  (N, K)
            x_S_x = (x_0[:, None, :] ** 2 * S_diag[None, :, :]).sum(dim=-1)

            # Stabilized computation: track O(1) part before scaling by 1/eps
            tilde_R_k = mu_x / T + x_S_x / (2.0 * T ** 2)
            log_weights = log_alpha[None, :] + tilde_R_k / eps  # (N, K)

            # shifted means:  μ̃_k = μ_k + Σ_k x_0 / T
            tilde_mu = mu[None, :, :] + S_diag[None, :, :] * x_0[:, None, :] / T

            # Avoid numerical issues with sqrt(0) when eps approaches zero
            scale_val = torch.sqrt(torch.clamp(eps * S_diag, min=1e-16))[None, :, :]

            mix = Categorical(logits=log_weights)
            comp = Independent(
                Normal(loc=tilde_mu, scale=scale_val),
                1,
            )
            gmm = MixtureSameFamily(mix, comp)
            return gmm.sample()

        else:  # general SPD
            S = self.get_Sigma()  # (K, D, D)

            # μ_k^T x_0  →  (N, K)
            mu_x = (mu[None, :, :] * x_0[:, None, :]).sum(dim=-1)

            # x_0^T Σ_k x_0  →  (N, K)
            x_col = x_0[:, None, :, None]                   # (N, 1, D, 1)
            x_S_x = (x_col.transpose(-1, -2) @ S[None] @ x_col)[:, :, 0, 0]

            # Stabilized computation
            tilde_R_k = mu_x / T + x_S_x / (2.0 * T ** 2)
            log_weights = log_alpha[None, :] + tilde_R_k / eps

            # μ̃_k = μ_k + Σ_k x_0 / T
            tilde_mu = (
                mu[None, :, :]
                + (S[None] @ x_0[:, None, :, None])[:, :, :, 0] / T
            )

            # Add a small diagonal ridge when covariance scales down to limit singularity
            cov_matrix = eps * S
            if eps < 1e-5:
                I_d = torch.eye(self.dim, device=x_0.device, dtype=x_0.dtype)
                cov_matrix = cov_matrix + 1e-8 * I_d[None, :, :]

            mix = Categorical(logits=log_weights)
            comp = MultivariateNormal(
                loc=tilde_mu,
                covariance_matrix=cov_matrix,
            )
            gmm = MixtureSameFamily(mix, comp)
            return gmm.sample()

    # ================================================================
    # Drift  σ_t u_{v_θ}(x, t)  via autograd  (Stabilized)
    # ================================================================

    def get_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the drift  σ_t u_{v_θ}(x, t) exactly as in Proposition 1.
        Numerically stabilized version avoiding O(1/eps) operators and matrix inversion.

        Parameters
        ----------
        x : (N, D)   current position
        t : (N,)     current time  (must satisfy 0 ≤ t < T)

        Returns
        -------
        drift : (N, D)
        """
        eps = self.epsilon
        T = self.T_horizon
        mu = self.get_mu()            # (K, D)
        log_alpha = self.log_alpha    # (K,)
        d = self.dim

        # make x a fresh leaf so autograd can differentiate w.r.t. it
        x = x.detach().requires_grad_(True)
        t_col = t[:, None]            # (N, 1)

        # ---- log N(x | 0, (T-t)ε I_d) ----   (N,)
        log_N_x = (
            -0.5 * d * torch.log(2 * math.pi * (T - t) * eps)
            - 0.5 * (x ** 2).sum(dim=-1) / ((T - t) * eps)
        )

        if self.is_diagonal:
            S_diag = self.get_Sigma_diag()        # (K, D)   = Σ_k (diagonal)
            S_inv_diag = 1.0 / S_diag             # (K, D)   = Σ_k^{-1}

            # 1. Use O(1) Stabilized Operators: tilde_A = eps * A, tilde_h = eps * h
            tilde_A_core = t_col / (T * (T - t_col))                    # (N, 1)
            tilde_A_k = tilde_A_core[:, :, None] + S_inv_diag[None, :, :] # (N, K, D)
            
            tilde_h_k = (x / (T - t_col))[:, None, :] + (S_inv_diag * mu)[None, :, :] # (N, K, D)

            # 2. Stable evaluation of A_k^{-1} h_k = tilde_A_k^{-1} tilde_h_k
            A_k_inv_h_k = tilde_h_k / tilde_A_k  # (N, K, D)

            # 3. Analytic log determinants (the log(eps) terms cancel out perfectly)
            log_det_tilde_A = torch.log(tilde_A_k).sum(dim=-1)          # (N, K)
            log_det_S = torch.log(S_diag).sum(dim=-1)                   # (K,)

            # 4. Pull out 1/eps factor from the exponents
            mu_Sinv_mu = (mu ** 2 / S_diag).sum(dim=-1)                 # (K,)
            quad_form = (tilde_h_k * A_k_inv_h_k).sum(dim=-1)           # (N, K)

            log_term_k = (
                log_alpha[None, :]
                - 0.5 * log_det_S[None, :]
                - 0.5 * log_det_tilde_A
                - 0.5 * (mu_Sinv_mu[None, :] - quad_form) / eps
            )

        else:  # general SPD covariance
            S_diag = self.get_Sigma_diag()        # (K, D)
            S_inv_diag = 1.0 / S_diag
            R = self.S_rotation                    # (K, D, D)

            S_inv = (R * S_inv_diag[:, None, :]) @ R.transpose(-1, -2)  # Σ_k^{-1}   (K, D, D)
            S_log_det = self.S_log_diag.sum(dim=-1)                     # log|Σ_k|   (K,)

            Sinv_mu = (S_inv @ mu[:, :, None])[:, :, 0]                 # Σ_k^{-1} μ_k  (K, D)

            # 1. Use O(1) Stabilized Operators
            scalar_term = t_col / (T * (T - t_col))                     # (N, 1)
            I_d = torch.eye(d, device=x.device, dtype=x.dtype)
            tilde_A_k = scalar_term[:, :, None, None] * I_d[None, None, :, :] + S_inv[None, :, :, :]
            tilde_h_k = (x / (T - t_col))[:, None, :] + Sinv_mu[None, :, :]

            # 2. Stable linear solve with well-conditioned O(1) matrix system
            A_k_inv_h_k = torch.linalg.solve(tilde_A_k, tilde_h_k)

            # 3. Analytic log determinants
            log_det_tilde_A = torch.logdet(tilde_A_k)

            # 4. Pull out 1/eps factor from the exponents
            mu_Sinv_mu = (mu[:, None, :] @ S_inv @ mu[:, :, None])[:, 0, 0]
            quad_form = (tilde_h_k[:, :, None, :] @ A_k_inv_h_k[:, :, :, None])[:, :, 0, 0]

            log_term_k = (
                log_alpha[None, :]
                - 0.5 * S_log_det[None, :]
                - 0.5 * log_det_tilde_A
                - 0.5 * (mu_Sinv_mu[None, :] - quad_form) / eps
            )

        # ---- Aggregating via logsumexp ----
        log_sum_k = torch.logsumexp(log_term_k, dim=-1)   # (N,)
        log_arg = log_N_x + log_sum_k   # (N,)

        # ---- σ_t u_{v_θ}(x,t) = ε ∇_x log_arg ----
        grad_log_arg = torch.autograd.grad(
            log_arg,
            x,
            grad_outputs=torch.ones_like(log_arg),
            create_graph=True,
        )[0]   # (N, D)

        return eps * grad_log_arg

    # ================================================================
    # Log-potential   log v_θ(x)
    # ================================================================

    def get_log_potential(self, x: torch.Tensor) -> torch.Tensor:
        """
        log v_θ(x_T) = log Σ_k α_k N(x_T | μ_k, ε Σ_k).

        Parameters
        ----------
        x : (N, D)

        Returns
        -------
        (N,)
        """
        eps = self.epsilon
        mu = self.get_mu()
        log_alpha = self.log_alpha

        if self.is_diagonal:
            S_diag = self.get_Sigma_diag()
            mix = Categorical(logits=log_alpha)
            comp = Independent(
                Normal(loc=mu, scale=torch.sqrt(eps * S_diag)), 1
            )
        else:
            S = self.get_Sigma()
            mix = Categorical(logits=log_alpha)
            comp = MultivariateNormal(
                loc=mu, covariance_matrix=eps * S
            )

        gmm = MixtureSameFamily(mix, comp)
        return gmm.log_prob(x) + torch.logsumexp(log_alpha, dim=-1)

    # ================================================================
    # Log normalising constant   log c_v(x_0) (Stabilized)
    # ================================================================

    def get_log_C(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        log c_v(x_0) = log Σ_k α_k exp(R_k(x_0)). Stabilized for small epsilon.

        Parameters
        ----------
        x_0 : (N, D)

        Returns
        -------
        (N,)
        """
        eps = self.epsilon
        T = self.T_horizon
        mu = self.get_mu()
        log_alpha = self.log_alpha

        # μ_k^T x_0   →  (N, K)
        mu_x = (mu[None, :, :] * x_0[:, None, :]).sum(dim=-1)

        if self.is_diagonal:
            S_diag = self.get_Sigma_diag()
            # x_0^T Σ_k x_0  →  (N, K)
            x_S_x = (x_0[:, None, :] ** 2 * S_diag[None, :, :]).sum(dim=-1)
        else:
            S = self.get_Sigma()
            x_col = x_0[:, None, :, None]
            x_S_x = (x_col.transpose(-1, -2) @ S[None] @ x_col)[:, :, 0, 0]

        # Scaled logsumexp configuration
        tilde_R_k = mu_x / T + x_S_x / (2.0 * T ** 2)
        return torch.logsumexp(log_alpha[None, :] + tilde_R_k / eps, dim=-1)

    # ================================================================
    # Inference Option 1:  Euler-Maruyama SDE simulation
    # ================================================================

    def sample_euler_maruyama(
        self, x: torch.Tensor, n_steps: int
    ) -> torch.Tensor:
        """
        Simulate the learned SDE via Euler-Maruyama.
        """
        eps = self.epsilon
        T = self.T_horizon
        dt = T / n_steps
        t = torch.zeros(x.shape[0], device=x.device)
        trajectory = [x]

        for _ in range(n_steps):
            drift = self.get_drift(x, t)
            noise = math.sqrt(float(dt * eps)) * torch.randn_like(x)
            x = x + drift * dt + noise
            t = t + dt
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)

    # ================================================================
    # Inference Option 2:  exact bridge trajectory sampling
    # ================================================================

    @torch.no_grad()
    def sample_bridge_trajectory(
        self, x_0: torch.Tensor, n_steps: int
    ) -> torch.Tensor:
        """
        Two-stage exact bridge trajectory (no SDE solver needed).
        """
        eps = self.epsilon
        T = self.T_horizon

        x_T = self.forward(x_0)

        dt = T / n_steps
        trajectory = [x_0]

        for l in range(1, n_steps):
            t_l = l * dt
            mean = (T - t_l) / T * x_0 + t_l / T * x_T
            std = math.sqrt(float(eps * t_l * (T - t_l) / T))
            x_t = mean + std * torch.randn_like(x_0)
            trajectory.append(x_t)

        trajectory.append(x_T)
        return torch.stack(trajectory, dim=1)

    # ================================================================
    # Convenience:  sample at a single time point
    # ================================================================

    @torch.no_grad()
    def sample_at_time(
        self, x_0: torch.Tensor, t_val: float
    ) -> torch.Tensor:
        """
        Sample x_t given x_0 using exact endpoint + bridge interpolation.
        """
        eps = self.epsilon
        T = self.T_horizon
        x_T = self.forward(x_0)

        mean = (T - t_val) / T * x_0 + t_val / T * x_T
        std = math.sqrt(float(eps * t_val * (T - t_val) / T))
        return mean + std * torch.randn_like(x_0)
