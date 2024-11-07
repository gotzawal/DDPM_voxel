# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, noise=None):
        B = x0.shape[0]
        timestep = torch.randint(0, self.var_scheduler.num_train_timesteps, (B,), device=self.device).long()

        if noise is None:
            noise = torch.randn_like(x0)

        x_noisy = self.var_scheduler.add_noise(x0, timestep, noise)

        predicted_noise = self.network(x_noisy, timestep)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def voxel_resolution(self):
        return self.network.voxel_resolution

    @torch.no_grad()
    def sample(self, batch_size):
        x = torch.randn(batch_size, 1, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution).to(self.device)
        for t in tqdm(reversed(range(self.var_scheduler.num_train_timesteps))):
            t_batch = torch.tensor([t] * batch_size, device=self.device)
            predicted_noise = self.network(x, t_batch)
            x = self.var_scheduler.step(x, t_batch, predicted_noise)
        return x

    def save(self, file_path):
        # state_dict만 저장
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        # state_dict만 로드
        self.load_state_dict(torch.load(file_path, map_location="cpu"))
