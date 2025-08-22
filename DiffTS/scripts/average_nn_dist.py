import os
import click
import open3d as o3d
import torch
from pykeops.torch import LazyTensor
import numpy as np
from tqdm import tqdm

def load_point_cloud_as_tensor(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        return None
    points = torch.tensor(pcd.points, dtype=torch.float32)
    return points

def distance_matrix_keops(pts1, pts2, device=torch.device("cuda")):
    K = 2
    pts1 = torch.tensor(pts1, device=device, dtype=torch.float64)
    pts2 = torch.tensor(pts2, device=device, dtype=torch.float64)
    
    X_i = LazyTensor(pts1[:, None, :])  # (10000, 1, 784) test set
    X_j = LazyTensor(pts2[None, :, :])  # (1, 60000, 784) train set
    # D_ij = ((X_i - X_j) ** 2).sum(-1)  # (10000, 60000) symbolic matrix of squared L2 distances
    D_ij = (X_i - X_j).norm(-1)
    dists, ind_knn = D_ij.Kmin_argKmin(K, dim=1) #.sum_reduction(dim=1)  # Samples <-> Dataset, (N_test, K)
    return dists.cpu().numpy(), ind_knn.cpu().numpy()

def compute_avg_nn_distance_torch(points):
    """
    Computes the average nearest-neighbor distance using PyTorch on GPU.
    points: (N, 3) torch tensor
    """
    if points.shape[0] < 2:
        return 0.0

    # Compute pairwise distances (N, N)
    diff = points.unsqueeze(1) - points.unsqueeze(0)
    dists = torch.sum(diff ** 2, dim=-1)

    # Exclude self-distance by setting diagonal to large number
    dists += torch.eye(points.shape[0], device=points.device) * 1e9

    # Get nearest neighbor distance for each point
    min_dists, _ = torch.min(dists, dim=1)

    return torch.mean(torch.sqrt(min_dists)).item()

@click.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def process_folder(folder):
    """Compute average nearest neighbor distance (on GPU using PyTorch) for each .ply file."""
    ply_files = [f for f in os.listdir(folder)]
    results = []

    if not ply_files:
        click.echo("No .ply files found in the specified folder.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")

    for ply_file in tqdm(ply_files):
        full_path = os.path.join(folder, ply_file)
        click.echo(f"Processing {ply_file}...")

        points = np.asarray(o3d.io.read_point_cloud(full_path).points)

        if points is None or points.shape[0] < 2:
            click.echo(f"  Warning: {ply_file} has insufficient points.")
            results[ply_file] = 0.0
            continue
        dists = distance_matrix_keops(points, points)
        dists = dists[0][:, 1]
        results.append(np.mean(dists))
    mean_dist = np.mean(results)
    print(f"Mean distance: {mean_dist:.4f} meters")

if __name__ == "__main__":
    process_folder()
