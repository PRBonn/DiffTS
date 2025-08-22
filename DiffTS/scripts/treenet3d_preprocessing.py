import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
import torch
from tree_utils.cloud import Cloud
from tree_utils.tree import TreeSkeleton
from tree_utils.branch import BranchSkeleton
from typing import Tuple

from copy import deepcopy
from pykeops.torch import LazyTensor

tree_size_correntions = { "shanshu":1,
                          "zhangshu":0.5,
                          "FloodedGum":0.5,
                          "Lemon":0.33,}

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

def generate_tree(tree_id, data: dict) -> Tuple[Cloud, TreeSkeleton]:
    branch_id = data[:, -1]
    branch_parent_id = data[:, -2]
    skeleton_xyz = data[:, :3].astype(np.float32)
    skeleton_radii = data[:, 3].astype(np.float32)    
    
    branch_uniques = np.unique(branch_id)
    
    branches = {}
    for branch in branch_uniques:
        # sort points based on distance to parent branch
        branch_mask = branch_id == branch
        # parent_id = branch_parent_id[branch_mask][0]
        # find parent id cause treenet is screeeeewed
        if branch == 1:
            parent_id = 0
            attach_point = skeleton_xyz[np.argmin(skeleton_xyz[branch_mask][:, 2])]
        else:
            # compute branch end points by computing covariance matrix
            branch_points = skeleton_xyz[branch_mask]
            if len(branch_points) == 1:
                end1 = 0
                end2 = 0
            elif len(branch_points) == 2:
                end1 = 0
                end2 = 1
            else:
                cov_matrix = np.cov(branch_points.T)
                try:
                    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
                except:
                    import ipdb;ipdb.set_trace()  # fmt: skip
                # find the eigenvector with the biggest eigenvalue
                main_axis = eig_vectors[:, np.argmax(eig_values)]
                # find the point with the smallest and the biggest projection on the main axis
                end1 = np.argmin(branch_points.dot(main_axis))
                end2 = np.argmax(branch_points.dot(main_axis))

            # compute distance of all points to all branch points
            branch_ends = branch_points[[end1,end2]]
            dists = np.linalg.norm(skeleton_xyz[:, None, :] - branch_ends[None, :, :], axis=-1).min(axis=1)
            # set all points with parent_id bigger than branch id to big distance
            dists[branch_id>=branch] = 1e6
            parent_id = branch_id[np.argmin(dists)]
            
            
            dist_to_parent = np.linalg.norm(branch_ends[:, None, :] - skeleton_xyz[branch_id == parent_id][None, :, :], axis=-1)
            try:
                dist_to_parent = dist_to_parent.min(axis=1)
            except:
                import ipdb;ipdb.set_trace()  # fmt: skip
            attach_point_id = np.argmin(dist_to_parent)
            attach_point = branch_ends[attach_point_id]
        
        attach_point_viz = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        attach_point_viz.translate(attach_point)
        
        dist_to_attach_point = np.linalg.norm(skeleton_xyz[branch_mask] - attach_point, axis=-1)
        sorted_ids = np.argsort(dist_to_attach_point)
        branch_skeleton_xyz = skeleton_xyz[branch_mask][sorted_ids]
        branch_skeleton_radii = skeleton_radii[branch_mask][sorted_ids]
        branches[branch] = BranchSkeleton(
            branch, parent_id, branch_skeleton_xyz, branch_skeleton_radii
        )
    skeleton = TreeSkeleton(tree_id, branches)
    return skeleton

def distance_matrix_keops(pts1, pts2, device=torch.device("cuda")):
    K = 1
    pts1 = torch.tensor(pts1, device=device, dtype=torch.float64)
    pts2 = torch.tensor(pts2, device=device, dtype=torch.float64)
    
    X_i = LazyTensor(pts1[:, None, :])  # (10000, 1, 784) test set
    X_j = LazyTensor(pts2[None, :, :])  # (1, 60000, 784) train set
    D_ij = (X_i - X_j).norm(-1)
    dists, ind_knn = D_ij.Kmin_argKmin(K, dim=1) #.sum_reduction(dim=1)  # Samples <-> Dataset, (N_t>
    return dists.cpu().numpy(), ind_knn.cpu().numpy()


def main(source_dir, output_dir, voxel_size=0.4):
    if not os.path.exists(os.path.join(output_dir, "split_db.npy")):
        print("Creating split database...")
        # fix seed
        np.random.seed(42)
        source_files = []
        train_files = []
        val_files = []
        test_files = []
        for variety_dir in os.listdir(source_dir):
            if not os.path.isdir(os.path.join(source_dir, variety_dir)):
                continue
            if variety_dir == "ULS":
                continue
            for subdir_variety in os.listdir(os.path.join(source_dir, variety_dir)):
                if not os.path.isdir(os.path.join(source_dir, variety_dir, subdir_variety)):
                    continue
                if "TLS" in subdir_variety:
                    continue
                source_files = [os.path.join(source_dir, variety_dir, subdir_variety, f) for f in os.listdir(os.path.join(source_dir, variety_dir, subdir_variety))]
                # randomize the order of the files
                np.random.shuffle(source_files)
                n_files = len(source_files)
                n_train = int(n_files * 0.8)
                n_val = int(n_files * 0.1)
                n_test = n_files - n_train - n_val
                train_files += source_files[:n_train]
                val_files += source_files[n_train:n_train+n_val]
                test_files += source_files[n_train+n_val:]    
        split_db = {"train": train_files, "val": val_files, "test": test_files}
        # save the split database
        np.save(os.path.join(output_dir, "split_db.npy"), split_db)
    else:
        split_db = np.load(os.path.join(output_dir, "split_db.npy"), allow_pickle=True).item()
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        file_list = split_db[split]
        for source_file in tqdm(file_list[::10]):
            output_file = os.path.join(output_dir, split, os.path.basename(source_file) + ".pt")
            if not os.path.isdir(source_file):
                continue
            if os.path.exists(output_file):
                continue
            
            source_branch_pcd = o3d.io.read_point_cloud(os.path.join(source_file, "pointcloud", "branchPointClouds.ply"))
            source_leaf_pcd = o3d.io.read_point_cloud(os.path.join(source_file, "pointcloud", "leafPointClouds.ply"))
            source_pcd = source_branch_pcd + source_leaf_pcd
            skeleton_raw = np.loadtxt(os.path.join(source_file, "skeleton", "skeleton.txt"))
            
            pcd_labels = np.zeros(len(source_pcd.points))
            pcd_labels[len(source_branch_pcd.points):] = 1
            
            points = np.asarray(source_pcd.points)
            
            # add 1 to all branch ids to start from 1
            skeleton_raw[:, -1] += 1
            skeleton_raw[:, -2] += 1
            
            if len(skeleton_raw) == 0:
                continue
            try:
                skeleton_points = skeleton_raw[:, :3]
            except:
                import ipdb;ipdb.set_trace()  # fmt: skip
            skeleton_pcd = o3d.geometry.PointCloud()
            skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_points)
            skeleton_pcd.paint_uniform_color([1, 0, 0])
            ids = np.where(skeleton_raw[:, -1] == 87)[0]
            np.asarray(skeleton_pcd.colors)[ids] = [0, 1, 0]
            skeleton = generate_tree(tree_id=os.path.basename(source_file), data=skeleton_raw)
            skeleton = repair_skeleton(skeleton)
            scan_point_colors = pcd_labels

            num_sample_points = -1
            try:
                highres_skeleton_nodes, highres_skeleton_edges, branch_id_mask, branch_glob_ids = upsample_skeleton(skeleton, 0.01)
            except:
                continue
            _, indices = distance_matrix_keops(np.asarray(source_pcd.points), highres_skeleton_nodes.numpy())
            medial_vecs = highres_skeleton_nodes[indices[:,0]] - np.asarray(source_pcd.points)


            data_item = {"scan_points": points, "scan_point_classes": pcd_labels, "skeleton": skeleton, "highres_skeleton_nodes": highres_skeleton_nodes, "highres_skeleton_edges": highres_skeleton_edges, "branch_id_mask": branch_id_mask, "branch_glob_ids": branch_glob_ids, "num_sample_points": num_sample_points, "scan_point_colors": scan_point_colors, "medial_vecs": medial_vecs}

            torch.save(data_item, output_file)
            

if __name__ == "__main__":
    source_dir = "/home/elias/workstation_data/datasets/TreeNet3D"
    output_dir = "/home/elias/workstation_data/datasets/TreeNet3D_preprocessed_st22"
    main(source_dir, output_dir)
