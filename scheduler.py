# scheduler.py
import torch
import torch.nn as nn
import numpy as np


class BaseScheduler(nn.Module):
    def __init__(self, num_train_timesteps, beta_1, beta_T, mode="linear"):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def add_noise(self, x0, t, noise):
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None, None]
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise


class DDPMScheduler(BaseScheduler):
    def step(self, x_t, t, eps_theta):
        betas_t = self.betas[t].view(-1, 1, 1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)
        pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * eps_theta) / torch.sqrt(alphas_cumprod_t)
        mean = torch.sqrt(alphas_t) * pred_x0 + torch.sqrt(1 - alphas_t) * eps_theta

        if t[0] > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        return mean + torch.sqrt(self.betas[t]).view(-1, 1, 1, 1, 1) * noise
