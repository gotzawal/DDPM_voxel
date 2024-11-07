# test_dataset.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# dataset.py에서 정의한 클래스와 함수 임포트
from dataset import VoxelDataset, get_voxel_dataloader  # 실제 파일 구조에 맞게 수정

def visualize_voxel_slices(voxel_grid, step, save_dir, prefix=''):
    """
    3D Voxel 데이터를 시각화하는 함수입니다.
    voxel_grid: (C, D, H, W) 형태의 torch.Tensor
    step: 현재 단계 (훈련 스텝 또는 샘플 번호)
    save_dir: 이미지 저장 디렉토리 (Path 객체)
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

def main():
    # 설정
    voxel_resolution = 64  # dataset.py에서 사용한 해상도와 일치
    voxel_data_path = './data/hdf5_data/chair_voxels_train.npy'  # 실제 데이터 경로로 수정
    batch_size = 4
    num_workers = 4
    save_dir = Path('./dataset_visualizations')
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 데이터로더 생성
    train_dl = get_voxel_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        voxel_data_path=voxel_data_path,
        voxel_resolution=voxel_resolution
    )
    
    # 데이터셋 정보 출력
    dataset = train_dl.dataset
    print(f"Dataset size: {len(dataset)}")
    
    # 첫 3개 샘플 확인
    for i in range(3):
        voxel = dataset[i]
        print(f"\nSample {i} Info:")
        print(f"Shape: {voxel.shape}")  # Expected: (1, 64, 64, 64)
        print(f"Dtype: {voxel.dtype}")  # Expected: torch.float32
        print(f"Min value: {voxel.min().item()}")
        print(f"Max value: {voxel.max().item()}")
        
        # NaN 및 Inf 값 확인
        has_nan = torch.isnan(voxel).any().item()
        has_inf = torch.isinf(voxel).any().item()
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Inf: {has_inf}")
        
        # 시각화
        visualize_voxel_slices(voxel, step=i, save_dir=save_dir, prefix='sample_')
    
    # 전체 데이터셋 통계 정보 출력
    print("\nCalculating dataset statistics...")
    all_data = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)  # Shape: (N, 1, D, H, W)
    print(f"All data shape: {all_data.shape}")
    print(f"Dtype: {all_data.dtype}")
    print(f"Min value: {all_data.min().item()}")
    print(f"Max value: {all_data.max().item()}")
    print(f"Mean value: {all_data.mean().item():.4f}")
    print(f"Std deviation: {all_data.std().item():.4f}")
    
    # NaN 및 Inf 값 확인
    dataset_has_nan = torch.isnan(all_data).any().item()
    dataset_has_inf = torch.isinf(all_data).any().item()
    print(f"Dataset contains NaN: {dataset_has_nan}")
    print(f"Dataset contains Inf: {dataset_has_inf}")
    
    # 배치 단위로 몇 개의 배치를 확인하고 시각화
    print("\nChecking batches...")
    for batch_idx, batch in enumerate(train_dl):
        print(f"\nBatch {batch_idx} Info:")
        print(f"Shape: {batch.shape}")  # Expected: (batch_size, 1, 64, 64, 64)
        print(f"Dtype: {batch.dtype}")
        print(f"Min value: {batch.min().item()}")
        print(f"Max value: {batch.max().item()}")
        
        # NaN 및 Inf 값 확인
        batch_has_nan = torch.isnan(batch).any().item()
        batch_has_inf = torch.isinf(batch).any().item()
        print(f"Contains NaN: {batch_has_nan}")
        print(f"Contains Inf: {batch_has_inf}")
        
        # 첫 샘플 시각화
        visualize_voxel_slices(batch[0], step=f'batch_{batch_idx}', save_dir=save_dir, prefix='batch_')
        
        if batch_idx >= 2:  # 첫 3배치만 확인
            break
    
    print("\nDataset testing completed. Visualizations saved in './dataset_visualizations/'.")

if __name__ == "__main__":
    main()
