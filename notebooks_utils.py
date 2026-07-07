import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
import scipy
from sklearn import datasets
import wandb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm

import sys

sys.path.append('..')
from src.discrete_ot import OTPlanSampler

# from eot_benchmark.metrics import compute_BW_UVP_by_gt_samples, calculate_cond_bw
# from eot_benchmark.gaussian_mixture_benchmark import (
#     get_guassian_mixture_benchmark_sampler,
#     get_guassian_mixture_benchmark_ground_truth_sampler, 
#     get_test_input_samples
# )


def mmd(x, y):
    Kxx = pairwise_distances(x, x)
    Kyy = pairwise_distances(y, y)
    Kxy = pairwise_distances(x, y)

    m = x.shape[0]
    n = y.shape[0]
    
    c1 = 1 / ( m * (m - 1))
    A = np.sum(Kxx - np.diag(np.diagonal(Kxx)))

    # Term II
    c2 = 1 / (n * (n - 1))
    B = np.sum(Kyy - np.diag(np.diagonal(Kyy)))

    # Term III
    c3 = 1 / (m * n)
    C = np.sum(Kxy)

    # estimate MMD
    mmd_est = -0.5*c1*A - 0.5*c2*B + c3*C
    
    return mmd_est


class TensorSampler:
    def __init__(self, tensor, device='cuda'):
        self.device = device
        self.tensor = torch.clone(tensor).to(device)
        
    def sample(self, size=5):
        assert size <= self.tensor.shape[0]
        
        ind = torch.tensor(np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False), device=self.device)
        return torch.clone(self.tensor[ind]).detach().to(self.device)

    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h
    

class StandardNormalSampler:
    def __init__(self, dim=1, device='cpu'):
        self.device = device
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)
    

class SwissRollSampler:
    def __init__(
        self, dim=2, device='cpu'
    ):
        self.device = device
        assert dim == 2
        self.dim = 2
        
    def sample(self, batch_size=10):
        batch = datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)
    

def fig2data(fig):
    """Convert a Matplotlib figure to an (H, W, 4) RGBA numpy array."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


def fig2img(fig):
    buf = fig2data(fig)
    h, w, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tobytes())


def pca_plot(x_0_gt, x_1_gt, x_1_pred, n_plot, save_name='plot_pca_samples.png', is_wandb=False):

    x_0_gt, x_1_gt, x_1_pred = x_0_gt.cpu(), x_1_gt.cpu(), x_1_pred.cpu()
    x_0_gt = x_0_gt[:n_plot]
    x_1_gt = x_1_gt[:n_plot]
    x_1_pred = x_1_pred[:n_plot]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=True, sharex=True, sharey=True)

    # For 2-D data, plot raw coordinates (PCA is unnecessary for Swiss roll).
    if x_1_gt.shape[1] == 2:
        x_0_plot = x_0_gt.numpy()
        x_1_plot = x_1_gt.numpy()
        x_1_pred_plot = x_1_pred.numpy()
    else:
        pca = PCA(n_components=2).fit(
            torch.cat([x_0_gt, x_1_gt, x_1_pred], dim=0).numpy()
        )
        x_0_plot = pca.transform(x_0_gt.numpy())
        x_1_plot = pca.transform(x_1_gt.numpy())
        x_1_pred_plot = pca.transform(x_1_pred.numpy())

    axes[0].scatter(
        x_0_plot[:, 0], x_0_plot[:, 1], c="g", edgecolor="black",
        label=r"$x\sim P_0(x)$", s=30,
    )
    axes[1].scatter(
        x_1_plot[:, 0], x_1_plot[:, 1], c="orange", edgecolor="black",
        label=r"$y\sim P_1(y)$", s=30,
    )
    axes[2].scatter(
        x_1_pred_plot[:, 0], x_1_pred_plot[:, 1], c="yellow", edgecolor="black",
        label=r"$y\sim \pi(y | x)$", s=30,
    )

    all_pts = np.vstack([x_0_plot, x_1_plot, x_1_pred_plot])
    margin = 0.1 * np.maximum(all_pts.max(axis=0) - all_pts.min(axis=0), 1e-6)
    x_lo, x_hi = all_pts[:, 0].min() - margin[0], all_pts[:, 0].max() + margin[0]
    y_lo, y_hi = all_pts[:, 1].min() - margin[1], all_pts[:, 1].max() + margin[1]

    for i in range(3):
        axes[i].grid()
        axes[i].set_xlim([x_lo, x_hi])
        axes[i].set_ylim([y_lo, y_hi])
        axes[i].legend()

    fig.tight_layout(pad=0.5)
    im = fig2img(fig)
    im.save(save_name)

    if is_wandb:
        wandb.log({f'Plot PCA samples': [wandb.Image(fig2img(fig))]})
        
        
def get_indepedent_plan_sample_fn(sampler_x, sampler_y):
    
    def ret_fn(batch_size):
        x_samples = sampler_x(batch_size)
        y_samples = sampler_y(batch_size)
        return x_samples, y_samples
    
    return ret_fn
     
    
def get_discrete_ot_plan_sample_fn(sampler_x, sampler_y, device='cpu'):
    
    ot_plan_sampler = OTPlanSampler('exact')
    
    def ret_fn(batch_size):
        
        x_samples = sampler_x(batch_size).to(device)
        y_samples = sampler_y(batch_size).to(device)
        
        return ot_plan_sampler.sample_plan(x_samples, y_samples)
    
    return ret_fn


class EOTGMMSampler:
    def __init__(self, dim, eps, batch_size=512, download=False) -> None:
        eps = eps if int(eps) < 1 else int(eps)

        self.dim = dim
        self.eps = eps
        self.x_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="input", dim=dim, eps=eps,
                                               batch_size=batch_size, device=f"cpu", download=download)
        self.y_sampler = get_guassian_mixture_benchmark_sampler(input_or_target="target", dim=dim, eps=eps,
                                                    batch_size=batch_size, device=f"cpu", download=download)
        self.gt_sampler = get_guassian_mixture_benchmark_ground_truth_sampler(dim=dim, eps=eps, 
                                                                        batch_size=batch_size,  device=f"cpu", download=download)
    
    def x_sample(self, batch_size):
        return self.x_sampler.sample(batch_size)
    
    def y_sample(self, batch_size):
        return self.y_sampler.sample(batch_size)
    
    def conditional_y_sample(self, x):
        return self.gt_sampler.conditional_plan.sample(x)

def get_gt_plan_sample_fn_EOT(eot_sampler: EOTGMMSampler):
    
    def ret_fn(batch_size):
        x_samples = eot_sampler.x_sample(batch_size)
        y_samples = eot_sampler.conditional_y_sample(x_samples)
        return x_samples, y_samples
    
    return ret_fn


def calcuate_condBW(model, dim, eps, n_samples=1000, device='cpu'):
    test_samples = get_test_input_samples(dim=dim, device=device)
    
    NUM_SAMPLES_METRICS = 1000

    model_input = test_samples.reshape(1000, 1, -1).repeat(1, NUM_SAMPLES_METRICS, 1).to(device)
    predictions = []

    with torch.no_grad():
        for test_samples_repeated in tqdm(model_input):
            predictions.append(model(test_samples_repeated).to(device))

    predictions = torch.stack(predictions, dim=0)
    
    cond_bw = calculate_cond_bw(test_samples, predictions, eps=eps, dim=dim)
    
    return cond_bw

