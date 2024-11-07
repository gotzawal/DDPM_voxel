# visualize.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_voxel_slices(voxel_grid, step, save_dir, prefix=''):
    """
    3D Voxel 데이터를 시각화하는 함수입니다.
    voxel_grid: (C, D, H, W) 형태의 torch.Tensor
    step: 현재 훈련 스텝
    save_dir: 이미지 저장 디렉토리
    prefix: 파일 이름에 추가할 접두사
    """
    voxel = voxel_grid.squeeze(0).cpu().numpy()  # (D, H, W)
    
    # 중앙 슬라이스 추출
    mid_d = voxel.shape[0] // 2
    mid_h = voxel.shape[1] // 2
    mid_w = voxel.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(voxel[mid_d, :, :], cmap='gray')
    axes[0].set_title('Mid D Slice')
    axes[0].axis('off')
    
    axes[1].imshow(voxel[:, mid_h, :], cmap='gray')
    axes[1].set_title('Mid H Slice')
    axes[1].axis('off')
    
    axes[2].imshow(voxel[:, :, mid_w], cmap='gray')
    axes[2].set_title('Mid W Slice')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}voxel_slices_step_{step}.png")
    plt.close()
