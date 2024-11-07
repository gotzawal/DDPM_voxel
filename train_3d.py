# train_3d.py
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset_3d import get_voxel_dataloader3D
from dotmap import DotMap
from model_3d import DiffusionModule3D
from network_3d import UNet3D
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from tqdm import tqdm
import numpy as np

matplotlib.use("Agg")

def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

def main(args):
    """Config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    now = get_current_time()
    if args.use_cfg:
        save_dir = Path(f"results/cfg_diffusion-3d-{now}")
    else:
        save_dir = Path(f"results/diffusion-3d-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config.toDict(), f, indent=2)

    voxel_resolution = config.voxel_resolution
    dataloader = get_voxel_dataloader3D(
        batch_size=config.batch_size,
        num_workers=4,
        voxel_data_path=config.voxel_data_path,
        voxel_resolution=voxel_resolution
    )

    # Set up the scheduler
    var_scheduler = DDPMScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode="linear",
    )

    network = UNet3D(
        T=config.num_diffusion_train_timesteps,
        voxel_resolution=voxel_resolution,
        ch=config.ch,
        ch_mult=config.ch_mult,
        attn=config.attn,
        num_res_blocks=config.num_res_blocks,
        dropout=config.dropout,
        use_cfg=config.use_cfg,
        cfg_dropout=config.cfg_dropout,
        num_classes=config.num_classes
    )

    ddpm = DiffusionModule3D(network, var_scheduler)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                ddpm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                samples = ddpm.sample(4, return_traj=False)
                # Visualization can be implemented as needed
                # For example, saving voxel grids as numpy files
                for i, sample in enumerate(samples.cpu().numpy()):
                    np.save(save_dir / f"step={step}-sample={i}.npy", sample.squeeze(0))
                    print(f"Saved voxel sample {i} at step {step}.")

                ddpm.save(f"{save_dir}/last.ckpt")
                ddpm.train()

            try:
                batch = next(iter(dataloader))
            except StopIteration:
                dataloader = get_voxel_dataloader3D(
                    batch_size=config.batch_size,
                    num_workers=4,
                    voxel_data_path=config.voxel_data_path,
                    voxel_resolution=voxel_resolution
                )
                batch = next(iter(dataloader))

            img = batch.to(config.device)  # Shape: (B, 1, D, H, W)
            # If using class labels, modify accordingly
            loss = ddpm.get_loss(img, class_label=None)  # Modify if using classes

            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--voxel_data_path",
        type=str,
        default='./data/hdf5_data/chair_voxels_train.npy',
        help="Path to the voxel data file.",
    )
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="Diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", type=int, nargs='+', default=[1, 2, 2, 2])
    parser.add_argument("--attn", type=int, nargs='+', default=[1])
    parser.add_argument("--num_res_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=None)  # Modify if using classes
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
