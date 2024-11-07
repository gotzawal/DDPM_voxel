# dataset_3d.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.ndimage import zoom

class VoxelDataset3D(Dataset):
    def __init__(self, voxel_data_path, voxel_resolution=64):
        super().__init__()
        self.voxel_data = np.load(voxel_data_path)  # Shape: (N, D, H, W)
        self.voxel_resolution = voxel_resolution

    def __len__(self):
        return len(self.voxel_data)

    def __getitem__(self, idx):
        voxel = self.voxel_data[idx]  # Shape: (D, H, W)

        # Resize to desired resolution using zoom
        if self.voxel_resolution != voxel.shape[0]:
            scale_factor = self.voxel_resolution / voxel.shape[0]
            voxel = zoom(voxel, zoom=scale_factor, order=0)

        # Convert to float32
        voxel = voxel.astype(np.float32)  # Already binary (0, 1)

        # Add channel dimension (C, D, H, W)
        voxel = np.expand_dims(voxel, axis=0)  # Shape: (1, D, H, W)

        # Convert to PyTorch tensor
        voxel_tensor = torch.from_numpy(voxel)

        return voxel_tensor

def get_voxel_dataloader3D(batch_size, num_workers=4, voxel_data_path='./data/hdf5_data/chair_voxels_train.npy', voxel_resolution=64):
    dataset = VoxelDataset3D(voxel_data_path, voxel_resolution=voxel_resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
