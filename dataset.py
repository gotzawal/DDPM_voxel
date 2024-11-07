# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.ndimage import zoom

class VoxelDataset(Dataset):
    def __init__(self, voxel_data_path, voxel_resolution=64):
        super().__init__()
        self.voxel_data = np.load(voxel_data_path)
        self.voxel_resolution = voxel_resolution

    def __len__(self):
        return len(self.voxel_data)

    def __getitem__(self, idx):
        voxel = self.voxel_data[idx]
        # 해상도 감소
        scale_factor = self.voxel_resolution / voxel.shape[0]
        voxel = zoom(voxel, zoom=scale_factor, order=1)
        # voxel 데이터를 float32 타입으로 변환하고 [-1, 1] 범위로 스케일링
        voxel = voxel.astype(np.float32) * 2.0 - 1.0
        # 채널 차원 추가 (C, D, H, W)
        voxel = np.expand_dims(voxel, axis=0)
        voxel = torch.from_numpy(voxel)
        return voxel

def get_voxel_dataloader(batch_size, num_workers=4, voxel_data_path='./data/hdf5_data/chair_voxels_train.npy', voxel_resolution=64):
    dataset = VoxelDataset(voxel_data_path, voxel_resolution=voxel_resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
