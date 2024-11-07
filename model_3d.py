# model_3d.py
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from network_3d import UNet3D

class DiffusionModule3D(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        B = x0.shape[0]

        # Sample random timesteps for each batch element
        timestep = torch.randint(0, self.var_scheduler.num_train_timesteps, (B,), device=self.device).long()

        # If noise is not provided, sample random noise
        if noise is None:
            noise = torch.randn_like(x0)

        # Add noise to x0 at the given timestep
        x_t, eps = self.var_scheduler.add_noise(x0, timestep, noise)

        # Predict the noise using the model's network
        predicted_noise = self.network(x_t, timestep, class_label)

        # Calculate loss (mean squared error between true and predicted noise)
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @property
    def device(self):
        return next(self.network.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
    ):
        x_T = torch.randn([batch_size, 1, self.network.voxel_resolution, self.network.voxel_resolution, self.network.voxel_resolution]).to(self.device)


        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            assert class_label is not None
            # Duplicate the batch for unconditional guidance
            class_label_uncond = torch.zeros_like(class_label)
            class_label = torch.cat([class_label_uncond, class_label], dim=0)
            guidance_scale = guidance_scale

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                B = x_t.shape[0] // 2
                x_t_uncond, x_t_cond = x_t.chunk(2, dim=0)
                noise_pred_uncond = self.network(x_t_uncond, t.to(self.device), class_label=None)
                noise_pred_cond = self.network(x_t_cond, t.to(self.device), class_label=class_label)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.network(x_t, t.to(self.device), class_label)

            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)

            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
