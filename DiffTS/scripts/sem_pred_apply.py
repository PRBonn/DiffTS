import click
import os
from tqdm import tqdm
import numpy as np
import torch

import open3d as o3d
from sklearn.neighbors import NearestNeighbors

@click.command()
@click.option('--sem_preds', '-i', type=str, help='Path to the input ply file', required=True)
@click.option('--pcds', '-i', type=str, help='Path to the input ply file', required=True)
@click.option('--output_path', '-o', type=str, help='Path to the output ply file', required=True)
def main(sem_preds, pcds, output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path+"_onlyBranches", exist_ok=True)
    varieties = ['Agarwood', 'ajianglanren', 'fenghuangmu', 'FloodedGum', 'Lemon', 'LombardyPoplar', 'shanshu', 'Tibetan_Cherry', 'xiaoyelanren', 'zhangshu']
    # load norm factors
    variety_max_xy_extent = np.load(os.path.join('/'.join(pcds.split('/')[:-1]), 'variety_max_xy_extent.npy'), allow_pickle=True).item()
    for file_name in tqdm(os.listdir(sem_preds)):
        if not file_name.endswith('.npz'):
            continue
        # skip if existing
        if os.path.exists(os.path.join(output_path, file_name.replace(".npz", ".pt"))):
            continue
        pred_sem = np.load(os.path.join(sem_preds, file_name))['pred_sem']
        
        data_item = torch.load(os.path.join(pcds, file_name.replace(".npz", ".pt")))
        cloud_offset = np.mean(data_item['scan_points'], axis=0).astype(np.float32)
        cloud_offset[2] = np.min(data_item['scan_points'][:,2]).astype(np.float32)
        max_val = variety_max_xy_extent[[v for v in varieties if v in file_name][0]]

        normed_scan_points = (data_item['scan_points'] - cloud_offset) / max_val
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.load(os.path.join(sem_preds, file_name))['coords'])
        distances, indices = nbrs.kneighbors(normed_scan_points)
        up_pred_sem = pred_sem[np.squeeze(indices)]
        
        data_item['pred_sem'] = up_pred_sem
        # torch.save(data_item, os.path.join(output_path, file_name.replace(".npz", ".pt")))
        
        # create pcd
        pcd = o3d.t.geometry.PointCloud()
        pcd.point['positions'] = o3d.core.Tensor(data_item['scan_points'], dtype=o3d.core.Dtype.Float32)
        pcd.point['semantics'] = o3d.core.Tensor(up_pred_sem[:, None], dtype=o3d.core.Dtype.UInt8)
        o3d.t.io.write_point_cloud(os.path.join(output_path, file_name.replace(".npz", ".ply")), pcd)
        
        # only branch pcd
        branch_pcd = o3d.t.geometry.PointCloud()
        branch_pcd.point['positions'] = o3d.core.Tensor(data_item['scan_points'][up_pred_sem == 0], dtype=o3d.core.Dtype.Float32)
        branch_pcd.point['semantics'] = o3d.core.Tensor(up_pred_sem[up_pred_sem == 0][:, None], dtype=o3d.core.Dtype.UInt8)
        # o3d.visualization.draw_geometries([branch_pcd.to_legacy()])
        o3d.t.io.write_point_cloud(os.path.join(output_path+"_onlyBranches", file_name.replace(".npz", ".xyz")), branch_pcd)
    
if __name__ == "__main__":
    main()