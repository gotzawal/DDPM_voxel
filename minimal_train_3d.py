# minimal_train_3d.py
import torch
from dataset_3d import VoxelDataset3D
from network_3d import UNet3D
from model_3d import DiffusionModule3D
from scheduler import DDPMScheduler

# Configuration
batch_size = 1
voxel_resolution = 32
num_diffusion_train_timesteps = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Dataset and DataLoader
dataset = VoxelDataset3D('./data/hdf5_data/chair_voxels_train.npy', voxel_resolution=voxel_resolution)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Initialize Scheduler
var_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_train_timesteps,
    beta_1=1e-4,
    beta_T=0.02,
    mode="linear",
)

# Initialize Network and Diffusion Module
network = UNet3D(
    T=num_diffusion_train_timesteps,
    voxel_resolution=voxel_resolution,
    ch=32,
    ch_mult=[1, 2],
    attn=[1],
    num_res_blocks=2,
    dropout=0.1,
    use_cfg=False,
    cfg_dropout=0.1,
    num_classes=None
)

ddpm = DiffusionModule3D(network, var_scheduler).to(device)

# Optimizer
optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)

# Training Step
try:
    for batch_idx, batch in enumerate(dataloader):
        img = batch.to(device)  # Shape: (B, 1, D, H, W)
        loss = ddpm.get_loss(img, class_label=None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        if batch_idx >= 10:
            break  # Run only a few batches for testing
except Exception as e:
    print(f"An error occurred during training: {e}")
