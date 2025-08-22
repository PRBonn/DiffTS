import glob
import os
from copy import deepcopy
from typing import Tuple

import click
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData
from pykeops.torch import LazyTensor
from tqdm import tqdm
from tree_utils.branch import BranchSkeleton
from tree_utils.cloud import Cloud
from tree_utils.tree import TreeSkeleton


def upsample_skeleton(st_skeleton: TreeSkeleton, sample_rate):
    ls = st_skeleton.to_o3d_lineset_with_branch_ids()
    branch_ids = [asdf[1] for asdf in ls]
    sampled_points = []
    sampled_ids = []
    branch_glob_ids = {}
    cumul_branch_offset = 0
    for curr_branch, branch_id in ls:
        edges = np.asarray(curr_branch.lines)
        xyz = np.asarray(curr_branch.points)

        pts, parent_id, node_id = [], [], []

        for i, edge in enumerate(edges):

            start = xyz[edge[0]]
            end = xyz[edge[1]]

            v = end - start
            length = np.linalg.norm(v)
            direction = v / length
            num_pts = int(np.round(length / sample_rate))

            if num_pts > 0:
                spaced_points = np.linspace(0, length, num_pts).reshape(-1, 1)
                pts.append(start + direction * spaced_points)
                node_ids = np.arange(num_pts) + cumul_branch_offset
                node_id.append(node_ids)
                cumul_branch_offset += num_pts
        if len(node_id) == 0:
            print("zero length branch", branch_id)
            continue      
        branch_glob_ids[branch_id] = np.concatenate(node_id, axis=0)    
        sampled_points.append(np.concatenate(pts, axis=0))
        sampled_ids.append(np.full(np.concatenate(pts, axis=0).shape[0], branch_id))
    node_pts = np.concatenate(sampled_points, axis=0)
    edge_list = []
    
    for branch in branch_glob_ids.keys():
        branch_parent_ids = branch_glob_ids[branch] - 1
        parent_id = st_skeleton.branches[branch].parent_id
        if parent_id not in branch_glob_ids:
            continue
        if parent_id != 0:
            try:
                closest_idx = np.sqrt((node_pts[branch_glob_ids[parent_id]] - node_pts[branch_glob_ids[branch][0]])**2).sum(-1).argmin()
            except:
                import ipdb;ipdb.set_trace()
                
            branch_parent_ids[0] = branch_glob_ids[parent_id][closest_idx]
        else:
            branch_parent_ids[0] = branch_glob_ids[branch][0]
        edge_list.append(np.stack([branch_parent_ids, branch_glob_ids[branch]], axis=1))
    
    dense_edges = np.concatenate(edge_list, axis=0)
    
    # create branch id mask
    branch_id_mask = np.zeros(node_pts.shape[0], dtype=np.int32)
    for branch in branch_glob_ids.keys():
        branch_id_mask[branch_glob_ids[branch]] = branch
    return torch.tensor(node_pts, dtype=torch.float), torch.tensor(dense_edges, dtype=torch.float), branch_id_mask, branch_glob_ids



def repair_skeleton(skeleton: TreeSkeleton):
    """ By default the skeletons are not connected between branches.
        this function connects the branches to their parent branches by finding
        the nearest point on the parent branch - relative to radius. 
        It returns a new skeleton with no reference to the original.
    """
    skeleton = deepcopy(skeleton)

    for branch in list(skeleton.branches.values()):

        if branch.parent_id == -1 or branch.parent_id == 0:
            continue
        
        # if branch.parent_id not in skeleton.branches:
        #     continue
        parent_branch = skeleton.branches[branch.parent_id]

        if len(branch.xyz) > 1:
            connection_pt, connection_rad = parent_branch.closest_pt_branchDir(
                pts=branch.xyz[[0,1]])
        else:
            connection_pt, connection_rad = parent_branch.closest_pt(
                pt=branch.xyz[[0]])

        branch.xyz = np.insert(branch.xyz, 0, connection_pt, axis=0)
        branch.radii = np.insert(branch.radii, 0, connection_rad, axis=0)

    return skeleton

def generate_tree(tree_id, vertices, edges) -> Tuple[Cloud, TreeSkeleton]:
    branch_id = np.arange(len(edges)) + 1
    branch_parent_id = np.full(len(edges), -1)
    for i, edge in enumerate(edges):
        parent_id = np.where(edges[:, 1] == edge[0])[0]
        if len(parent_id) > 0:
            branch_parent_id[i] = parent_id 
        else:
            # take closest node as parent
            dists = np.linalg.norm(vertices[edges[:,1]] - vertices[edge[0]], axis=-1)
            dists[i] = np.inf
            if dists.min() < 1e-6:
                branch_parent_id[i] = np.argmin(dists)
    branch_parent_id += 1
    skeleton_xyz = vertices[edges].reshape(-1,3)
    skeleton_radii = np.zeros(len(skeleton_xyz))
    sizes = np.full(len(edges),2)

    offsets = np.cumsum(np.append([0], sizes))

    branch_idx = [np.arange(size) + offset for size, offset in zip(sizes, offsets)]
    branches = {}

    for idx, _id, parent_id in zip(branch_idx, branch_id, branch_parent_id):  
        branches[_id] = BranchSkeleton(
            _id, parent_id, skeleton_xyz[idx], skeleton_radii[idx]
        )

    return TreeSkeleton(tree_id, branches)

def nn_keops(pts1, pts2, device=torch.device("cuda")):
    K = 1
    pts1 = torch.tensor(pts1, device=device, dtype=torch.float64)
    pts2 = torch.tensor(pts2, device=device, dtype=torch.float64)
    
    X_i = LazyTensor(pts1[:, None, :])  # (10000, 1, 784) test set
    X_j = LazyTensor(pts2[None, :, :])  # (1, 60000, 784) train set
    D_ij = (X_i - X_j).norm(-1)
    dists, ind_knn = D_ij.Kmin_argKmin(K, dim=1) #.sum_reduction(dim=1)  # Samples <-> Dataset, (N_test, K)
    return dists.cpu().numpy(), ind_knn.cpu().numpy()

@click.command()
@click.option("--source_dir", "-s", type=str, help="Path to the source directory", required=True)
@click.option("--gt_dir", "-g", type=str, help="Path to the ground truth directory", required=True)
@click.option("--output_dir", "-o", type=str, help="Path to the output directory", required=True)
@click.option("--filter_threshold", "-ft", type=float, help="Threshold for filtering edges", default=0.05)
def main(source_dir, gt_dir, output_dir, filter_threshold, voxel_size=0.4):
    split_db = np.load(os.path.join(output_dir, "split_db.npy"), allow_pickle=True).item()
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        if split == "val" or split == "test":
            os.makedirs(os.path.join(output_dir, split+"_pcd"), exist_ok=True)
        file_list = split_db[split]
        for source_file in tqdm(file_list):
            output_file = os.path.join(output_dir, split, source_file.split('/')[-2].split('_')[0]+'_'+os.path.basename(source_file).replace(".ply", ".pt"))
            if not source_file.endswith(".ply"):
                continue
            if 'skeleton' in source_file:
                continue
            source_pcd = o3d.t.io.read_point_cloud(source_file)
            
            ply = PlyData.read(source_file.replace(".ply", "_skeleton.ply"))
            vertices = np.vstack((ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z'])).T
            edges = np.vstack(ply['edge']['vertex_indices'])
            if filter_threshold > 0:
                edge_lineset = o3d.geometry.LineSet()
                edge_lineset.points = o3d.utility.Vector3dVector(vertices)
                edge_lineset.lines = o3d.utility.Vector2iVector(edges)
                edge_lineset.paint_uniform_color([1,0,0])
                # filter based on distance to scan
                edge_vertex_pcd = o3d.geometry.PointCloud()
                edge_vertex_pcd.points = o3d.utility.Vector3dVector(vertices[edges].reshape(-1,3))
                dists = edge_vertex_pcd.compute_point_cloud_distance(source_pcd.to_legacy())
                dists = np.array(dists).reshape(-1,2)
                vertex_mask = (dists < filter_threshold).any(axis=-1)
                
                edges = edges[vertex_mask]
                filtered_edge_lineset = o3d.geometry.LineSet()
                filtered_edge_lineset.points = o3d.utility.Vector3dVector(vertices)
                filtered_edge_lineset.lines = o3d.utility.Vector2iVector(edges)
                filtered_edge_lineset.paint_uniform_color([0,0,1])
            
            skeleton = generate_tree(tree_id=os.path.basename(source_file), vertices=vertices, edges=edges)
            origin_viz = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            origin_viz.translate(vertices[0])
            
            try:
                highres_skeleton_nodes, highres_skeleton_edges, branch_id_mask, branch_glob_ids = upsample_skeleton(skeleton, 0.005)
            except:
                continue
            _, indices = nn_keops(source_pcd.point.positions.numpy(), highres_skeleton_nodes.numpy())
            medial_vecs = highres_skeleton_nodes[indices[:,0]] - source_pcd.point.positions.numpy()
            
            # visualize as lineset
            viz_lineset = o3d.geometry.LineSet()
            viz_lineset.points = o3d.utility.Vector3dVector(np.concatenate([source_pcd.point.positions.numpy(), source_pcd.point.positions.numpy()+medial_vecs.numpy()], axis=0))
            # lines are from source to highres
            viz_lineset.lines = o3d.utility.Vector2iVector(np.stack([np.arange(len(source_pcd.point.positions.numpy())),np.arange(len(source_pcd.point.positions.numpy())) + len(source_pcd.point.positions.numpy())], axis=1))
            viz_pcd = o3d.geometry.PointCloud()
            viz_pcd.points = o3d.utility.Vector3dVector(source_pcd.point.positions.numpy())
            highres_pcd = o3d.geometry.PointCloud()
            highres_pcd.points = o3d.utility.Vector3dVector(highres_skeleton_nodes.numpy())
            highres_pcd.paint_uniform_color([1,0,0])
            o3d.visualization.draw_geometries([viz_lineset, viz_pcd, highres_pcd])
            data_item = {"scan_points": source_pcd.point.positions.numpy(), "skeleton": skeleton,  "scan_point_colors": source_pcd.point.colors.numpy(), "medial_vecs": medial_vecs}
            torch.save(data_item, output_file)
            
            

if __name__ == "__main__":
    main()
