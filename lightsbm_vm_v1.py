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

        self.register_buffer("epsilon", torch.tensor(float(epsilon)))
        self.register_buffer("T_horizon", torch.tensor(float(T)))

        self.log_alpha = nn.Parameter(
            torch.log(torch.ones(n_potentials) / n_potentials)
        )

        self.mu = nn.Parameter(torch.randn(n_potentials, dim))

        self.S_log_diag = nn.Parameter(
            torch.log(S_diagonal_init * torch.ones(n_potentials, dim))
        )

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

    def init_mu_by_samples(self, samples: torch.Tensor):
        assert samples.shape[0] == self.n_potentials
        self.mu.data = samples.clone().to(self.mu.device)

    init_r_by_samples = init_mu_by_samples

    def set_epsilon(self, new_epsilon: float):

        self.epsilon = torch.tensor(float(new_epsilon), device=self.epsilon.device)

    def get_Sigma_diag(self) -> torch.Tensor:
        return torch.exp(self.S_log_diag)

    def get_Sigma(self) -> torch.Tensor:
        S_diag = self.get_Sigma_diag()
        if self.is_diagonal:
            return S_diag
        else:
            R = self.S_rotation  # (K, D, D)
            return (R * S_diag[:, None, :]) @ R.transpose(-1, -2)

    def get_mu(self) -> torch.Tensor:
        return self.mu

    @torch.no_grad()
    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        eps = self.epsilon
        T = self.T_horizon
        mu = self.get_mu()            
        log_alpha = self.log_alpha   

        if self.is_diagonal:
            S_diag = self.get_Sigma_diag()  

            mu_x = (mu[None, :, :] * x_0[:, None, :]).sum(dim=-1)

            x_S_x = (x_0[:, None, :] ** 2 * S_diag[None, :, :]).sum(dim=-1)

            tilde_R_k = mu_x / T + x_S_x / (2.0 * T ** 2)
            log_weights = log_alpha[None, :] + tilde_R_k / eps  

            tilde_mu = mu[None, :, :] + S_diag[None, :, :] * x_0[:, None, :] / T


            scale_val = torch.sqrt(torch.clamp(eps * S_diag, min=1e-16))[None, :, :]

            mix = Categorical(logits=log_weights)
            comp = Independent(
                Normal(loc=tilde_mu, scale=scale_val),
                1,
            )
            gmm = MixtureSameFamily(mix, comp)
            return gmm.sample()

        else:
            S = self.get_Sigma()
        
            mu_x = (mu[None, :, :] * x_0[:, None, :]).sum(dim=-1)

            x_col = x_0[:, None, :, None]
            x_S_x = (x_col.transpose(-1, -2) @ S[None] @ x_col)[:, :, 0, 0]

            tilde_R_k = mu_x / T + x_S_x / (2.0 * T ** 2)
            log_weights = log_alpha[None, :] + tilde_R_k / eps

            tilde_mu = (
                mu[None, :, :]
                + (S[None] @ x_0[:, None, :, None])[:, :, :, 0] / T
            )

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

    def get_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps = self.epsilon
        T = self.T_horizon
        mu = self.get_mu()            
        log_alpha = self.log_alpha    
        d = self.dim

        x = x.detach().requires_grad_(True)
        t_col = t[:, None]            

        log_N_x = (
            -0.5 * d * torch.log(2 * math.pi * (T - t) * eps)
            - 0.5 * (x ** 2).sum(dim=-1) / ((T - t) * eps)
        )

        if self.is_diagonal:
            S_diag = self.get_Sigma_diag()        
            S_inv_diag = 1.0 / S_diag             

            tilde_A_core = t_col / (T * (T - t_col))                    
            tilde_A_k = tilde_A_core[:, :, None] + S_inv_diag[None, :, :] 
            
            tilde_h_k = (x / (T - t_col))[:, None, :] + (S_inv_diag * mu)[None, :, :] 

            A_k_inv_h_k = tilde_h_k / tilde_A_k
            
            log_det_tilde_A = torch.log(tilde_A_k).sum(dim=-1)         
            log_det_S = torch.log(S_diag).sum(dim=-1)                  

            mu_Sinv_mu = (mu ** 2 / S_diag).sum(dim=-1)
            quad_form = (tilde_h_k * A_k_inv_h_k).sum(dim=-1)           

            log_term_k = (
                log_alpha[None, :]
                - 0.5 * log_det_S[None, :]
                - 0.5 * log_det_tilde_A
                - 0.5 * (mu_Sinv_mu[None, :] - quad_form) / eps
            )

        else:  
            S_diag = self.get_Sigma_diag()        
            S_inv_diag = 1.0 / S_diag
            R = self.S_rotation                    

            S_inv = (R * S_inv_diag[:, None, :]) @ R.transpose(-1, -2) 
            S_log_det = self.S_log_diag.sum(dim=-1)                     

            Sinv_mu = (S_inv @ mu[:, :, None])[:, :, 0]
            
            scalar_term = t_col / (T * (T - t_col))                    
            I_d = torch.eye(d, device=x.device, dtype=x.dtype)
            tilde_A_k = scalar_term[:, :, None, None] * I_d[None, None, :, :] + S_inv[None, :, :, :]
            tilde_h_k = (x / (T - t_col))[:, None, :] + Sinv_mu[None, :, :]

            A_k_inv_h_k = torch.linalg.solve(tilde_A_k, tilde_h_k)

            log_det_tilde_A = torch.logdet(tilde_A_k)

            mu_Sinv_mu = (mu[:, None, :] @ S_inv @ mu[:, :, None])[:, 0, 0]
            quad_form = (tilde_h_k[:, :, None, :] @ A_k_inv_h_k[:, :, :, None])[:, :, 0, 0]

            log_term_k = (
                log_alpha[None, :]
                - 0.5 * S_log_det[None, :]
                - 0.5 * log_det_tilde_A
                - 0.5 * (mu_Sinv_mu[None, :] - quad_form) / eps
            )

        log_sum_k = torch.logsumexp(log_term_k, dim=-1)   
        log_arg = log_N_x + log_sum_k
        
        grad_log_arg = torch.autograd.grad(
            log_arg,
            x,
            grad_outputs=torch.ones_like(log_arg),
            create_graph=True,
        )[0]   

        return eps * grad_log_arg

    def get_log_potential(self, x: torch.Tensor) -> torch.Tensor:
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

    def get_log_C(self, x_0: torch.Tensor) -> torch.Tensor:
        eps = self.epsilon
        T = self.T_horizon
        mu = self.get_mu()
        log_alpha = self.log_alpha

        mu_x = (mu[None, :, :] * x_0[:, None, :]).sum(dim=-1)

        if self.is_diagonal:
            S_diag = self.get_Sigma_diag()
            x_S_x = (x_0[:, None, :] ** 2 * S_diag[None, :, :]).sum(dim=-1)
        else:
            S = self.get_Sigma()
            x_col = x_0[:, None, :, None]
            x_S_x = (x_col.transpose(-1, -2) @ S[None] @ x_col)[:, :, 0, 0]

        tilde_R_k = mu_x / T + x_S_x / (2.0 * T ** 2)
        return torch.logsumexp(log_alpha[None, :] + tilde_R_k / eps, dim=-1)

    def sample_euler_maruyama(
        self, x: torch.Tensor, n_steps: int
    ) -> torch.Tensor:
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

    @torch.no_grad()
    def sample_bridge_trajectory(
        self, x_0: torch.Tensor, n_steps: int
    ) -> torch.Tensor:
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

    @torch.no_grad()
    def sample_at_time(
        self, x_0: torch.Tensor, t_val: float
    ) -> torch.Tensor:
        eps = self.epsilon
        T = self.T_horizon
        x_T = self.forward(x_0)

        mean = (T - t_val) / T * x_0 + t_val / T * x_T
        std = math.sqrt(float(eps * t_val * (T - t_val) / T))
        return mean + std * torch.randn_like(x_0)
