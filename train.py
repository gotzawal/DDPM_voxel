# train.py
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import random
import numpy as np
from dataset import get_voxel_dataloader
from model import DiffusionModule
from network import UNet
from scheduler import DDPMScheduler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from visualize import visualize_voxel_slices  # 시각화 함수 임포트

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

def main(args):
    """config"""
    config = vars(args)
    config['device'] = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    now = get_current_time()
    save_dir = Path(f"results/voxel_diffusion-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(args.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    voxel_resolution = args.voxel_resolution
    train_dl = get_voxel_dataloader(
        batch_size=args.batch_size,
        num_workers=4,
        voxel_data_path=args.voxel_data_path,
        voxel_resolution=voxel_resolution
    )
    train_it = iter(train_dl)

    var_scheduler = DDPMScheduler(
        args.num_diffusion_train_timesteps,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        mode="linear",
    )

    network = UNet(
        T=args.num_diffusion_train_timesteps,
        voxel_resolution=voxel_resolution,
        ch=32,
        ch_mult=[1, 2, 2, 4],
        attn=[],  # Attention 레이어 제거
        num_res_blocks=1,
        dropout=0.0,
    )

    ddpm = DiffusionModule(network, var_scheduler)
    ddpm = ddpm.to(config['device'])

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / args.warmup_steps, 1.0)
    )

    scaler = GradScaler()

    step = 0
    losses = []

    # 훈련 시작 전에 몇 개의 원본 데이터 시각화
    print("Visualizing original training data...")
    try:
        original_voxel = next(iter(train_dl)).to(config['device'])
        visualize_voxel_slices(original_voxel[0], step='original', save_dir=save_dir, prefix='original_')
    except Exception as e:
        print(f"Error visualizing original data: {e}")

    with tqdm(total=args.train_num_steps) as pbar:
        while step < args.train_num_steps:
            if step % args.log_interval == 0 and step != 0:
                ddpm.eval()
                # 샘플 생성 및 시각화
                try:
                    with torch.no_grad():
                        sample = ddpm.sample(batch_size=1)
                    visualize_voxel_slices(sample[0], step=step, save_dir=save_dir, prefix='sample_')
                except Exception as e:
                    print(f"Error during sampling: {e}")
                # 모델 저장
                ddpm.save(f"{save_dir}/last.ckpt")
                ddpm.train()

            try:
                voxel = next(train_it)
            except StopIteration:
                train_it = iter(train_dl)
                voxel = next(train_it)
            voxel = voxel.to(config['device'])
            with autocast():
                loss = ddpm.get_loss(voxel)
            
            # 손실이 NaN인지 확인
            if torch.isnan(loss):
                print(f"NaN loss encountered at step {step}. Exiting training.")
                break

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

            # 파라미터에 NaN이 있는지 확인
            nan_detected = False
            for name, param in ddpm.network.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN detected in parameter: {name}")
                    nan_detected = True
                    break
            if nan_detected:
                print("NaN detected in model parameters. Exiting training.")
                break

            # 훈련 중간에 노이즈가 잘 추가되었는지 시각화
            if step % args.log_interval == 0 and step != 0:
                try:
                    with torch.no_grad():
                        # 원본 데이터
                        original = voxel[0].cpu()
                        visualize_voxel_slices(original, step=f"original_step_{step}", save_dir=save_dir, prefix='original_')
                        # 노이즈가 추가된 데이터
                        B = voxel.shape[0]
                        timestep = torch.randint(0, var_scheduler.num_train_timesteps, (B,), device=config['device']).long()
                        noise = torch.randn_like(voxel)
                        x_noisy = var_scheduler.add_noise(voxel, timestep, noise)
                        visualize_voxel_slices(x_noisy[0], step=f"noisy_step_{step}", save_dir=save_dir, prefix='noisy_')
                except Exception as e:
                    print(f"Error during intermediate visualization: {e}")

            pbar.set_description(f"Loss: {loss.item():.4f}")
            step += 1
            pbar.update(1)

    # 손실 곡선 저장
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title("Training Loss Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_dir / "loss_curve.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_num_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--num_diffusion_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voxel_data_path", type=str, default="./data/hdf5_data/chair_voxels_train.npy")
    parser.add_argument("--voxel_resolution", type=int, default=64)
    args = parser.parse_args()
    main(args)
