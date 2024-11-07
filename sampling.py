# sampling.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import DiffusionModule
from network import UNet
from scheduler import DDPMScheduler

# 시드 설정 함수 정의
import random
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 모델 생성 (훈련 시와 동일한 설정)
voxel_resolution = 64  # 훈련 시 사용한 해상도와 동일하게 설정
var_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_1=1e-4,
    beta_T=0.02,
    mode="linear",
)
network = UNet(
    T=1000,
    voxel_resolution=voxel_resolution,
    ch=32,  # 채널 수 감소
    ch_mult=[1, 2, 2, 4],
    attn=[],  # Attention 레이어 제거
    num_res_blocks=1,  # Residual Block 수 감소
    dropout=0.0,  # Dropout 제거
)

# Diffusion 모델 생성
ddpm = DiffusionModule(network, var_scheduler)
ddpm = ddpm.to(device)

# 모델 로드
checkpoint_path = 'path/to/your/last.ckpt'  # 실제 모델 파일 경로로 변경
ddpm.load(checkpoint_path)
ddpm.eval()

# 샘플 생성
with torch.no_grad():
    samples = ddpm.sample(batch_size=1)  # 필요한 만큼의 샘플 수로 변경 가능

print("Generated sample shape:", samples.shape)
print("Sample data range:", samples.min().item(), samples.max().item())

def visualize_voxel(voxel_grid, threshold=0.0, title="Voxel"):
    """
    3D Voxel 데이터를 시각화하는 함수입니다.
    voxel_grid: (C, D, H, W) 형태의 torch.Tensor
    threshold: voxel이 채워진 것으로 간주될 임계값
    """
    # NaN 값을 0으로 대체
    voxel_grid = torch.nan_to_num(voxel_grid, nan=0.0)
    voxel_grid = (voxel_grid > threshold).squeeze().cpu().numpy()
    filled = np.argwhere(voxel_grid)

    if filled.size == 0:
        print(f"No voxels are filled with the threshold {threshold}.")
        return

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(filled[:, 0], filled[:, 1], filled[:, 2], zdir='z', c='red', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    max_range = np.array([filled[:, 0].max() - filled[:, 0].min(),
                          filled[:, 1].max() - filled[:, 1].min(),
                          filled[:, 2].max() - filled[:, 2].min()]).max() / 2.0

    mid_x = (filled[:, 0].max() + filled[:, 0].min()) * 0.5
    mid_y = (filled[:, 1].max() + filled[:, 1].min()) * 0.5
    mid_z = (filled[:, 2].max() + filled[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.savefig(f"{title}.png")  # 시각화된 이미지를 저장
    plt.close()

# 샘플 시각화 (임계값을 낮춰 시도)
visualize_voxel(samples[0], threshold=-1.0, title="sample_threshold_-1.0")
visualize_voxel(samples[0], threshold=-0.5, title="sample_threshold_-0.5")
visualize_voxel(samples[0], threshold=0.0, title="sample_threshold_0.0")
