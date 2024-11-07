import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_voxel_from_npy(file_path):
    """
    Visualizes a 3D binary voxel grid from a .npy file.

    Parameters:
    file_path (str): Path to the .npy file containing the 3D voxel data.
    """
    # Load voxel data from .npy file
    voxel_grid = np.load(file_path)
    
    # Ensure the voxel grid is binary (0s and 1s)
    voxel_grid = (voxel_grid > 0.5).astype(int)  # Threshold to ensure binary values if needed
    
    # Identify occupied voxel positions
    occupied_voxels = np.argwhere(voxel_grid == 1)
    
    if occupied_voxels.size == 0:
        print("No voxels are filled in this sample.")
        return
    
    # Plot the occupied voxels
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        occupied_voxels[:, 0], 
        occupied_voxels[:, 2], 
        occupied_voxels[:, 1], 
        c='red', marker='o'
    )
    
    # Set labels and aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_box_aspect([1, 1, 1])
    
    # Set limits based on voxel grid dimensions
    ax.set_xlim([0, voxel_grid.shape[0]])
    ax.set_ylim([0, voxel_grid.shape[1]])
    ax.set_zlim([0, voxel_grid.shape[2]])
    
    # Turn off the axes
    ax.axis("off")
    
    plt.title("Voxel Grid Visualization")
    plt.show()

# Example usage
file_path = './results/diffusion-3d-11-07-204358/step=38000-sample=0.npy'  # Replace with the actual path to your .npy file
visualize_voxel_from_npy(file_path)