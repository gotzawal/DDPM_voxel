# sampling_3d.py
import argparse
import numpy as np
import torch
from pathlib import Path
from model_3d import DiffusionModule3D
from network_3d import UNet3D
from scheduler import DDPMScheduler
from dataset_3d import VoxelDataset3D  # If needed for class labels

def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Initialize the network
    network = UNet3D(
        T=args.num_diffusion_train_timesteps,
        voxel_resolution=args.voxel_resolution,
        ch=args.ch,
        ch_mult=args.ch_mult,
        attn=args.attn,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=args.num_classes
    )

    # Initialize the scheduler
    var_scheduler = DDPMScheduler(
        args.num_diffusion_train_timesteps,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        mode="linear",
    )

    # Initialize the diffusion module
    ddpm = DiffusionModule3D(network, var_scheduler)
    ddpm.load(args.ckpt_path)
    ddpm.eval()
    ddpm = ddpm.to(device)

    total_num_samples = args.total_num_samples
    batch_size = args.batch_size
    num_batches = int(np.ceil(total_num_samples / batch_size))

    for i in range(num_batches):
        sidx = i * batch_size
        eidx = min(sidx + batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:
            # Implement class conditioning if necessary
            # For simplicity, assuming unconditional here
            samples = ddpm.sample(
                B,
                class_label=None,  # Replace with actual labels if using
                guidance_scale=args.cfg_scale
            )
        else:
            samples = ddpm.sample(B)

        samples = samples.cpu().numpy()
        for j in range(B):
            np.save(save_dir / f"sample_{sidx + j}.npy", samples[j, 0])  # Save single channel
            print(f"Saved sample {sidx + j}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_diffusion_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", type=int, nargs='+', default=[1, 2, 2, 2])
    parser.add_argument("--attn", type=int, nargs='+', default=[1])
    parser.add_argument("--num_res_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=None)  # Modify if using classes
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--total_num_samples", type=int, default=500)
    args = parser.parse_args()
    main(args)
