import random

import fpsample
import numpy as np
import open3d as o3d
import torch
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

from DiffTS.smart_tree.data_types.tree import TreeSkeleton


def find_parent(sampled_points, sampled_branch_ids_all, parent_ids, branch):
    parent_id = parent_ids[branch]
    sampled_parent_ids = sampled_branch_ids_all == parent_id
    if sampled_parent_ids.sum() == 0:
        if parent_id == 0:
            return []
        return find_parent(sampled_points, sampled_branch_ids_all, parent_ids, parent_id)
    return sampled_parent_ids

def graph_farthest_point_sampling(graph, num_samples):
    num_nodes = graph.shape[0]
    selected_nodes = []
    
    # Initialize the first node randomly
    initial_node = random.randint(0, num_nodes - 1)
    selected_nodes.append(initial_node)
    
    # Distance array to store the shortest distance to the closest selected node
    min_distances = np.full(num_nodes, np.inf)
    
    for _ in tqdm(range(num_samples - 1)):
        # Compute shortest path from the last added node to all other nodes
        distances = dijkstra(graph, directed=False, indices=selected_nodes[-1])
        
        # Update the minimum distances to any selected node
        min_distances = np.minimum(min_distances, distances)
        
        # Select the farthest node from the set of selected nodes
        farthest_node = np.argmax(min_distances)
        selected_nodes.append(farthest_node)
    
    return selected_nodes
    
def upsample_skeleton(st_skeleton: TreeSkeleton, sample_rate):
    ls = st_skeleton.to_o3d_lineset_with_branch_ids()
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
            continue      
        branch_glob_ids[branch_id] = np.concatenate(node_id, axis=0)    
        sampled_points.append(np.concatenate(pts, axis=0))
        sampled_ids.append(np.full(np.concatenate(pts, axis=0).shape[0], branch_id))
    node_pts = np.concatenate(sampled_points, axis=0)
    edge_list = []
    
    for branch in branch_glob_ids.keys():
        branch_parent_ids = branch_glob_ids[branch] - 1
        parent_id = st_skeleton.branches[branch].parent_id
        if parent_id != 0:
            closest_idx = np.sqrt((node_pts[branch_glob_ids[parent_id]] - node_pts[branch_glob_ids[branch][0]])**2).sum(-1).argmin()
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

def find_parent_legacy(sampled_points, sampled_branch_ids_all, st_skeleton, branch):
    parent_id = st_skeleton.branches[branch].parent_id
    # sampled_parent_ids = np.in1d(sampled_points, branch_glob_ids[parent_id])
    sampled_parent_ids = sampled_branch_ids_all == parent_id
    if sampled_parent_ids.sum() == 0:
        if parent_id == 0:
            return []
        return find_parent_legacy(sampled_points, sampled_branch_ids_all, st_skeleton, parent_id)
    return sampled_parent_ids

def downsample_skeleton_legacy(st_skeleton: TreeSkeleton, input_nodes, input_edges, branch_id_mask, branch_global_ids, num_sample_points=1000, farthest_node_sampling=False, debug=False):    
    if num_sample_points > input_nodes.shape[0]:
        print("num_sample_points is greater than the number of input nodes")
    
    if farthest_node_sampling:
        raise NotImplementedError("Farthest node sampling not implemented")
    else:
        if num_sample_points == input_nodes.shape[0]:
            sample_idx = np.arange(input_nodes.shape[0])
        if num_sample_points / input_nodes.shape[0] > 0.7:
            sample_idx = torch.randperm(input_nodes.shape[0])[:num_sample_points]
        else:
            try:
                sample_idx = fpsample.bucket_fps_kdline_sampling(input_nodes.numpy(), num_sample_points, h=9).astype(np.int64)
            except:
                import ipdb;ipdb.set_trace()  # fmt: skip
        sample_mask = torch.zeros(input_nodes.shape[0], dtype=torch.bool)
        sample_mask[sample_idx] = True
    sampled_points = torch.arange(input_nodes.shape[0])[sample_mask]
    sampled_branch_ids_all = branch_id_mask[sampled_points]
    high_res_to_low_res_id_mapping = torch.full((input_nodes.shape[0],), -1)
    try:
        high_res_to_low_res_id_mapping[sampled_points] = torch.arange(num_sample_points)
    except:
        print("sampled_points", sampled_points.shape, "num_sample_points", num_sample_points, input_nodes.shape)
        import ipdb;ipdb.set_trace()  # fmt: skip
    sampled_edge_list = []
    for branch in branch_global_ids.keys():
        sampled_branch_ids = sampled_points[sampled_branch_ids_all == branch]
        if len(sampled_branch_ids) == 0:
            continue
        branch_parent_ids = sampled_branch_ids.clone()
        branch_parent_ids[1:] = sampled_branch_ids[:-1]
        parent_id = st_skeleton.branches[branch].parent_id
        if parent_id != 0:
            sampled_parent_ids = find_parent_legacy(sampled_points, sampled_branch_ids_all, st_skeleton, branch)
            if len(sampled_parent_ids) == 0:
                branch_parent_ids[0] = sampled_branch_ids[0]
            else:
                sampled_parent_ids = sampled_points[sampled_parent_ids]
                first_node_curr_branch = input_nodes[branch_id_mask == branch][0]
                closest_idx = np.sqrt((input_nodes[sampled_parent_ids] - first_node_curr_branch)**2).sum(-1).argmin()
                branch_parent_ids[0] = sampled_parent_ids[closest_idx]
        else:
            branch_parent_ids[0] = sampled_branch_ids[0]
        sampled_edge_list.append(np.stack([high_res_to_low_res_id_mapping[branch_parent_ids], high_res_to_low_res_id_mapping[sampled_branch_ids]], axis=1))

    subs_nodes = input_nodes[sampled_points]
    sampled_edge_list = np.concatenate(sampled_edge_list, axis=0)
    
    if debug:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(subs_nodes)
        line_set.lines = o3d.utility.Vector2iVector(sampled_edge_list)
        node_pcd = o3d.geometry.PointCloud()
        node_pcd.points = o3d.utility.Vector3dVector(subs_nodes)
        node_pcd.paint_uniform_color([1,0,0])
        full_node_pcd = o3d.geometry.PointCloud()
        full_node_pcd.points = o3d.utility.Vector3dVector(input_nodes)
        full_node_pcd.paint_uniform_color([0,1,0])
        o3d.visualization.draw_geometries([line_set, node_pcd])
    assert (np.arange(sampled_edge_list.shape[0]) == sampled_edge_list[:,1]).all()
    node_parent_ids = torch.tensor(sampled_edge_list[:,0])
    return subs_nodes, node_parent_ids