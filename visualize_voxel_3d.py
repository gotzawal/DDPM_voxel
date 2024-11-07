# visualize_voxel_3d.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_voxel(voxel_grid, sample_idx):
    """
    Visualizes a 3D binary voxel grid using matplotlib.

    Parameters:
    voxel_grid (numpy.ndarray): A 3D binary voxel grid where 1 indicates occupancy and 0 indicates empty.
    sample_idx (int): Sample index for labeling.
    """
    occupied_voxels = np.argwhere(voxel_grid == 1)

    if occupied_voxels.size == 0:
        print(f"No voxels are filled in sample {sample_idx}.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(occupied_voxels[:, 0], occupied_voxels[:, 2], occupied_voxels[:, 1], c='red', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')  # Y와 Z 축을 바꿨습니다. 필요에 따라 조정 가능합니다.
    ax.set_zlabel('Y')

    ax.set_box_aspect([1,1,1])  # 등간격 축 비율

    ax.set_xlim([0, voxel_grid.shape[0]])
    ax.set_ylim([0, voxel_grid.shape[1]])
    ax.set_zlim([0, voxel_grid.shape[2]])

    ax.axis("off")

    plt.title(f"Sample {sample_idx}")
    plt.show()
