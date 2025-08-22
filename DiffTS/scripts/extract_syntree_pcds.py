import os
from DiffTS.tree_utils.file import load_data_npz
import open3d as o3d
folder = "/home/elias/workstation_data/datasets/synthetic-trees_apples/foliage"

for split in ['validation', 'test']:
    out_folder = split + '_pcds'
    out_folder = os.path.join(folder, out_folder)
    os.makedirs(out_folder, exist_ok=True)
    files = os.listdir(os.path.join(folder, split))
    for file in files:
        if file.endswith('.npz'):
            datapath = os.path.join(os.path.join(folder, split), file)
            scan_points_st, skeleton = load_data_npz(datapath)
            print(f'Processing {datapath}')
            out_path = os.path.join(out_folder, file.replace('.npz', '.xyz'))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scan_points_st.xyz)
            pcd = pcd.voxel_down_sample(voxel_size=0.03)  # Downsample for better visualization
            o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
            