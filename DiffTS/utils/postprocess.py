import numpy as np
import open3d as o3d
import torch
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


def meanGridSampling(
    pcd: torch.Tensor,
    resolution_meter,
    scale=1.0,
    features=None,
    smpl_feats=None,
    mask=None,
    fill_value=0,
):
    """Computes the mean over all points in the grid cells

    Args:
        pcd (torch.Tensor): [...,N,3] point coordinates
        resolution_meter ([type]): grid resolution in meters
        scale (float, optional): Defaults to 1.0. Scale to convert resolution_meter to grid_resolution: resolution = resolution_meter/scale
        features (torch.Tensor): [...,N,D] point features
        smpl_feats (torch.Tensor): [...,N,D] additional point features to be grid sampled
    Returns:
        grid_coords (torch.Tensor): [...,N,3] grid coordinates
        grid_features (torch.Tensor): [...,N,D] grid features
        grid_smpl_feats (torch.Tensor): [...,N,D] additional point features to have been grid sampled
        mask (torch.Tensor): [...,N,1] valid point mask (True: valid, False: not valid)
    """
    resolution = resolution_meter / scale
    if len(pcd.shape) < 3:
        pcd = pcd.unsqueeze(0)
    if len(features.shape) < 3:
        features = features.unsqueeze(0)
    B = pcd.shape[0]

    grid_coords = torch.zeros_like(pcd, device=pcd.device)
    grid_features = torch.zeros_like(features, device=pcd.device)
    if smpl_feats != None:
        if len(smpl_feats.shape) < 3:
            smpl_feats = smpl_feats.unsqueeze(0)
        grid_smpl_feats = torch.zeros_like(smpl_feats, device=pcd.device)
    out_mask = torch.full_like(pcd[..., :1], False, dtype=bool, device=pcd.device)

    if mask is not None:
        pcd[~mask.expand_as(pcd)] = float("inf")
    grid = torch.floor((pcd - pcd.min(dim=-2, keepdim=True)[0]) / resolution).double()

    if mask is not None:
        pcd[~mask.expand_as(pcd)] = fill_value
    if mask is not None:
       grid_size = grid[mask.squeeze(-1)].max().detach() + 1
    else:
       grid_size = grid.max().detach() + 1
    grid_idx = (
       grid[..., 0] + grid[..., 1] * grid_size + grid[..., 2] * grid_size * grid_size
   )

    max_nr = []
    for i in range(B):
        unique, indices, counts = torch.unique(
            grid_idx[i], return_inverse=True, dim=None, return_counts=True
        )
        indices.unsqueeze_(-1)

        nr_cells = len(counts)
        if unique[-1].isinf():
            counts = counts[:-1]
            nr_cells -= 1
        max_nr.append(nr_cells)

        grid_coords[i].scatter_add_(-2, indices.expand(pcd[i].shape), pcd[i])
        grid_coords[i, :nr_cells, :] /= counts.unsqueeze(-1)

        grid_features[i].scatter_add_(
            -2, indices.expand(features[i].shape), features[i]
        )
        grid_features[i, :nr_cells, :] /= counts.unsqueeze(-1)
        if smpl_feats != None:
            grid_smpl_feats[i].scatter_add_(
                -2, indices.expand(smpl_feats[i].shape), smpl_feats[i]
            )
            grid_smpl_feats[i, :nr_cells, :] /= counts.unsqueeze(-1)
        out_mask[i, :nr_cells, :] = True

        if fill_value != 0:
            grid_coords[i, nr_cells:] = fill_value

    max_nr = max(max_nr)
    grid_coords = grid_coords[..., :max_nr, :]
    grid_features = grid_features[..., :max_nr, :]
    out_mask = out_mask[..., :max_nr, :]
    if smpl_feats != None:
        grid_smpl_feats = grid_smpl_feats[..., :max_nr, :]
        return grid_coords, grid_features, out_mask, grid_smpl_feats
    else:
        return grid_coords, grid_features, out_mask

def visualize_mst_open3d(nodes, mst):
    points = o3d.utility.Vector3dVector(nodes)
    lines = []
    connected = mst.nonzero()
    lines = np.vstack((connected[0], connected[1])).T
    
    line_set = o3d.geometry.LineSet(
        points=points,
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color([0, 0, 1])
    return line_set

def min_spanning_tree(node_pred, cond_pts, pred_parent_pos, max_bridge_dist=0.2, filter_resolution=0.1, averaging=False, dir_min_weight=0.5, connected_components_filt=False, debug=False):
    if debug:
        input_pcd = o3d.geometry.PointCloud()
        input_pcd.points = o3d.utility.Vector3dVector(node_pred.cpu().detach().numpy())
        input_pcd.paint_uniform_color([1, 0, 0])
    # discard nodes far from conditioning points
    dists = torch.cdist(node_pred, cond_pts)
    dists = dists.min(dim=-1)[0]
    inlier_ids = torch.where(dists < max_bridge_dist)[0]
    # pcd_pred = pcd_pred.select_by_index(inlier_ids)
    node_pred = node_pred[inlier_ids]
    pred_parent_pos = pred_parent_pos[inlier_ids]
    
    node_dir_vecs = node_pred[None, :, :] - node_pred[:, None, :] # [N_nodes, N_nodes, 3]
    node_dir_vecs_n = node_dir_vecs / torch.linalg.norm(node_dir_vecs, dim=-1, keepdim=True) # [N_nodes (from), N_nodes (to), 3]
    pred_dir_vecs = pred_parent_pos - node_pred # [N_nodes, 3]
    pred_dir_vecs = pred_dir_vecs / torch.linalg.norm(pred_dir_vecs, dim=-1, keepdim=True) # [N_nodes, 3]
    dir_weights = node_dir_vecs_n * pred_dir_vecs[:, None, :] # [N_nodes (from), N_nodes (to), 3]
    dir_weights = dir_weights.sum(dim=-1) # [N_nodes (from), N_nodes (to)] 
    dir_weights[dir_weights.isnan()] = 0 # [N_nodes (from), N_nodes (to)] values from -1 to 1
    # remap [-1,1] to [1,0.5]
    dir_max_weight = 4.0
    # min max normalization    
    dir_weights = (dir_weights +1)/2*(dir_max_weight - dir_min_weight) + dir_min_weight

    dist_matrix = node_dir_vecs.norm(dim=-1) * dir_weights
    
    if debug:
        before_filt = visualize_mst_open3d(node_pred, dist_matrix).paint_uniform_color([1, 0, 0])
    
    threshold = 2 * max_bridge_dist #0.7
    dist_matrix[dist_matrix > threshold] = 0  # Apply threshold to distance matrix
    mst = minimum_spanning_tree(dist_matrix).toarray()
    
    if debug:
        mst_lineset = visualize_mst_open3d(node_pred, mst).paint_uniform_color([0, 1, 0])
    
    if connected_components_filt == True:
        # compute edge lenghts
        non_zero_edges = mst.nonzero()
        edge_lengths = np.sqrt((node_pred[non_zero_edges[0]] - node_pred[non_zero_edges[1]]) ** 2).sum(axis=1)
        # set edges longer than the threshold to 0
        outlier_mask = edge_lengths > max_bridge_dist
        outlier_edges = (non_zero_edges[0][outlier_mask], non_zero_edges[1][outlier_mask])
        mst[outlier_edges] = 0
        # filter by connected components
        min_n_nodes = 2
        n_componenents, component_labels = connected_components(mst)
        cluster_ids, cluster_size = np.unique(component_labels, return_counts=True)
        inlier_clusters = cluster_ids[cluster_size > min_n_nodes]
        outlier_clusters = cluster_ids[cluster_size <= min_n_nodes]
        inlier_mask = np.isin(component_labels, inlier_clusters)
        # in_outlier_mask = np.isin(component_labels, outlier_clusters)
        
        # inliers_connected_outliers = inlier_mask & in_outlier_mask
        mst[inlier_mask == False] = 0
        
        # pre_reconnect = visualize_mst_open3d(node_pred, mst)
        mst = dist_matrix.clone()
        mst[inlier_mask == False] = 0
        mst[:, inlier_mask == False] = 0
        # all nodes connected to removed nodes have to be reconnected to the closest node
        
        mst = minimum_spanning_tree(mst).toarray()
    
    graph_lineset = visualize_mst_open3d(node_pred, mst).paint_uniform_color([0, 1, 0])
    
    if debug:
        o3d.visualization.draw_geometries([input_pcd, graph_lineset])
    
    node_pred_pcd = o3d.geometry.PointCloud()
    node_pred_pcd.points = o3d.utility.Vector3dVector(node_pred.cpu().detach().numpy())
    node_pred_pcd.paint_uniform_color([0, 1, 0])
    # predicted offsets
    pred_flows = o3d.geometry.LineSet()
    pred_flows.points = o3d.utility.Vector3dVector(torch.cat((node_pred, pred_parent_pos), dim=0).cpu().detach().numpy())
    pred_flows.lines = o3d.utility.Vector2iVector(np.concatenate((np.arange(node_pred.shape[0])[:, None], np.arange(node_pred.shape[0], node_pred.shape[0] + pred_parent_pos.shape[0])[:, None]), axis=1))
    
    if debug:
        cond_pts_pcd = o3d.geometry.PointCloud()
        cond_pts_pcd.points = o3d.utility.Vector3dVector(cond_pts.cpu().detach().numpy())
        cond_pts_pcd.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([input_pcd, node_pred_pcd, graph_lineset, pred_flows])
        
    return graph_lineset