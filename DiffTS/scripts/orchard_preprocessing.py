import os
from copy import deepcopy
from functools import partial
from typing import Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData
from pycpd import DeformableRegistration
from pykeops.torch import LazyTensor
from scipy.sparse import coo_matrix, csr_array
from scipy.sparse.csgraph import (connected_components, depth_first_order,
                                  minimum_spanning_tree)
from tqdm import tqdm
from tree_utils.branch import BranchSkeleton
from tree_utils.cloud import Cloud
from tree_utils.tree import TreeSkeleton

from DiffTS.utils.postprocess import visualize_mst_open3d
from DiffTS.utils.pytimer import Timer

def upsample_lineset(skeleton, sample_dist):
    edges = np.asarray(skeleton.lines)
    vertices = np.asarray(skeleton.points)
    upsampled_vertices = []
    upsampled_edges = []
    vertex_offset = 0

    for edge in edges:
        start = vertices[edge[0]]
        end = vertices[edge[1]]

        vector = end - start
        length = np.linalg.norm(vector)
        direction = vector / length
        num_points = int(np.ceil(length / sample_dist))

        if num_points > 1:
            local_sample_dist = length / (num_points - 1)
            sampled_points = [start + direction * (i * local_sample_dist) for i in range(num_points)]
            upsampled_vertices.extend(sampled_points)

            # Create edges between consecutive sampled points
            for i in range(len(sampled_points) - 1):
                upsampled_edges.append([vertex_offset + i, vertex_offset + i + 1])

            vertex_offset += len(sampled_points)
        else:
            # If the edge is too short, just add the original edge
            upsampled_vertices.extend([start, end])
            upsampled_edges.append([vertex_offset, vertex_offset + 1])
            vertex_offset += 2

    # Convert to numpy arrays
    upsampled_vertices = np.array(upsampled_vertices)
    upsampled_edges = np.array(upsampled_edges)
    return upsampled_vertices, upsampled_edges

def precompute_sampled_ancestors(parent, is_sampled):
    N = len(parent)
    nearest = np.full(N, -1)
    for node in range(N):
        current = node
        while current != 0:
            current = parent[current]
            if is_sampled[current]:
                nearest[node] = current
                break
    return nearest

def filter_duplicate_vertices(vertices, edges):
    # filter duplicate vertices
    # compute edge lengths
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1)
    # remove all duplicate nodes, (edge length 0)
    for zero_edge in np.where(edge_lengths == 0)[0][::-1]:
        # remove the node from the edges
        removed_edge = edges[zero_edge]
        edges = np.delete(edges, zero_edge, axis=0)
        edges[:,0][edges[:,0] == removed_edge[1]] = removed_edge[0]
        edges[:,1][edges[:,1] == removed_edge[1]] = removed_edge[0]

    unique_verts, mapping = np.unique(edges.reshape(-1), return_inverse=True)
    vertices = vertices[unique_verts]
    edges = mapping.reshape(-1, 2)
    return vertices, edges

def reconnect_graph(graph, vertices, perform_minimum_spanning_tree=True):
    _, inverse_indices = np.unique(vertices, return_inverse=True, axis=0)
    
    same_group = inverse_indices[:, None] == inverse_indices[None, :]
    np.fill_diagonal(same_group, False)  # Exclude self-pairs
    graph[same_group] = 1e-6  # Set distance for same vertices to a small value
       
    if perform_minimum_spanning_tree:
        graph = minimum_spanning_tree(graph)
    root_node = 0
    _, predecessors = depth_first_order(graph, i_start=root_node, directed=False, return_predecessors=True)
    return predecessors

def downsample_ls(vertices, edges, n_samples):
    parents = np.concatenate((np.array((-1,)),edges[:, 0]))
    is_sampled = np.zeros(len(vertices), dtype=bool)
    is_sampled[:n_samples] = True
    np.random.shuffle(is_sampled)
    is_sampled[0] = True
    nearest = precompute_sampled_ancestors(parents, is_sampled)
    
    edges = np.stack((nearest[is_sampled], np.arange(len(vertices))[is_sampled]), axis=1)
    edges = edges[edges[:, 0] != -1]
    unique_verts, mapping = np.unique(edges.reshape(-1), return_inverse=True)
    vertices = vertices[unique_verts]
    upsampled_edges = mapping.reshape(-1, 2)
    parents = np.concatenate((np.array((-1,)), upsampled_edges[:, 0]))
    return vertices, parents, upsampled_edges

@click.command()
@click.option("--source_dir", "-s", type=str, help="Path to the source directory", required=True, multiple=True)
@click.option("--output_dir", "-o", type=str, help="Path to the output directory", required=True)
@click.option("--filter_threshold", "-ft", type=float, help="Threshold for filtering edges", default=0.05)
@click.option("--reference_folder", "-r", type=str, help="Path to the reference directory", required=True)
def main(source_dir, output_dir, filter_threshold, reference_folder, voxel_size=0.4):
    debug_timing = False
    tim = Timer()
    tim.tic()
    if not os.path.exists(os.path.join(output_dir, "split_db.npy")):
        print("Creating split database...")
        # fix seed
        np.random.seed(42)
        source_files = []
        train_files = []
        val_files = []
        test_files = []
        source_files = []
        for date_dir in source_dir:
            for lane_dir in os.listdir((date_dir)):
                source_files += [os.path.join(date_dir, lane_dir, f) for f in os.listdir(os.path.join(date_dir, lane_dir))]
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
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "split_db.npy"), split_db)
    else:
        split_db = np.load(os.path.join(output_dir, "split_db.npy"), allow_pickle=True).item()
    
    if debug_timing:
        tim.tocTic("Split database created in")
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
            print(f"Processing {source_file}...")
            source_pcd = o3d.t.io.read_point_cloud(source_file)
            lane = source_file.split('/')[-1].split('_')[-1].split('.')[0]
            reference_date = reference_folder.split('/')[-2]
            file_name = '_'.join(os.path.basename(source_file).split('_')[:2] + [reference_date] + [lane] + ["skeleton.ply"])
            skeleton_path = os.path.join(reference_folder, lane+"_tree_skeletons", file_name)
            ply = PlyData.read(skeleton_path)
            vertices = np.vstack((ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z'])).T
            edges = np.vstack(ply['edge']['vertex_indices'])
            if filter_threshold > 0:
                # filter based on distance to scan
                edge_vertex_pcd = o3d.geometry.PointCloud()
                edge_vertex_pcd.points = o3d.utility.Vector3dVector(vertices[edges].reshape(-1,3))
                dists = edge_vertex_pcd.compute_point_cloud_distance(source_pcd.to_legacy())
                dists = np.array(dists).reshape(-1,2)
                close_edge_mask = (dists < filter_threshold).any(axis=-1)
                # readd low vetices
                edge_vertices = np.asarray(edge_vertex_pcd.points)
                low_vertex_mask = edge_vertices[:, 2] < edge_vertices[:, 2].min() + 0.2
                low_vertex_mask = low_vertex_mask.reshape(-1, 2).any(axis=-1)
                close_edge_mask = np.logical_or(close_edge_mask, low_vertex_mask)
                
                # remove connected components with less than 10 nodes
                min_n_nodes = 10
                graph = csr_array((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices)))
                n_componenents, component_labels = connected_components(graph)
                cluster_ids, cluster_size = np.unique(component_labels, return_counts=True)
                big_cluster_mask = cluster_size > min_n_nodes
                
                # remove more clusters next to root
                cluster_centroids = np.array([vertices[component_labels == cid].mean(axis=0) for cid in cluster_ids])
                low_min_n_nodes = 50
                invalid_low_cluster_mask = (cluster_centroids[:, 2] < vertices[:, 2].min() + 0.3) & (cluster_size < low_min_n_nodes)
                inlier_clusters = cluster_ids[big_cluster_mask & ~invalid_low_cluster_mask]
                inlier_mask = np.isin(component_labels, inlier_clusters)
                inlier_ids = np.where(inlier_mask)[0]
                inlier_edge_ids = np.isin(edges, inlier_ids).any(axis=-1)
                inlier_edge_ids = np.logical_and(inlier_edge_ids, close_edge_mask)
                
                # update node inlier_ids to reflect also distance filtering
                inlier_ids = np.unique(edges[inlier_edge_ids])
                
                inlier_node_pcd = o3d.geometry.PointCloud()
                inlier_node_pcd.points = o3d.utility.Vector3dVector(vertices[inlier_ids])
                inlier_node_pcd.paint_uniform_color([0, 1, 0])
                
                filtered_edge_lineset = visualize_mst_open3d(vertices, graph)
                np.asarray(filtered_edge_lineset.colors)[inlier_edge_ids] = [0, 1, 0]  # green for inliers
                
                parents = reconnect_graph(graph, vertices)
                if debug_timing:
                    tim.tocTic("skeleton reconnect")
                on_path = np.zeros(len(vertices), dtype=bool)
                for node in inlier_ids:
                    while node != -9999 and not on_path[node]:
                        on_path[node] = True
                        node = parents[node]
                if debug_timing:
                    tim.tocTic("skeleton path finding")
                traversed_nodes = np.where(on_path)[0]
                if debug_timing:
                    tim.tocTic("skeleton path nodes")
                filtered_edge_lineset = visualize_mst_open3d(vertices, graph)
                if debug_timing:
                    tim.tocTic("skeleton graph to lineset")
                
                lines =  np.asarray(filtered_edge_lineset.lines)
                inlier_edges = np.isin(lines, traversed_nodes).any(axis=-1)
                np.asarray(filtered_edge_lineset.colors)[inlier_edges] = [0, 1, 0] 
                
                if debug_timing:
                    tim.tocTic("skeleton filter edges")
                edges = lines[inlier_edges]
                vertices, edges = filter_duplicate_vertices(vertices, edges)
                if debug_timing:
                    tim.tocTic("skeleton filter dups")
                
                # ensure graph is a tree
                graph = csr_array((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices)))
                _, parents = depth_first_order(graph, i_start=0, directed=False, return_predecessors=True)
                # reconstruct edges from parents
                edges = np.vstack([parents, np.arange(len(vertices))]).T[1:]
                if debug_timing:
                    tim.tocTic("skeleton reconnect after filtering")
            
            skeleton_ls = o3d.geometry.LineSet()
            skeleton_ls.points = o3d.utility.Vector3dVector(vertices)
            skeleton_ls.lines = o3d.utility.Vector2iVector(edges)
            skeleton_ls.paint_uniform_color([0, 1, 0])
            
            # remove scan point farther than 20cm from skeleton
            vertex_pcd = o3d.geometry.PointCloud()
            vertex_pcd.points = o3d.utility.Vector3dVector(vertices)
            dists = source_pcd.to_legacy().compute_point_cloud_distance(vertex_pcd)
            dists = np.array(dists)
            filtered_source_pcd = source_pcd.select_by_index(np.where(dists < 0.15)[0])
            
            if debug_timing:
                tim.tocTic("skeleton preprocessing")
            upsampled_vertices, upsampled_edges = upsample_lineset(skeleton_ls, sample_dist=0.002)
            if debug_timing:
                tim.tocTic("skeleton upsampling")
            graph = csr_array((np.ones(len(upsampled_edges)), (upsampled_edges[:, 0], upsampled_edges[:, 1])), shape=(len(upsampled_vertices), len(upsampled_vertices)))

            if debug_timing:
                tim.tocTic("graph init")
            parents = reconnect_graph(graph, upsampled_vertices, perform_minimum_spanning_tree=False)
            if debug_timing:
                tim.tocTic("skeleton reconnect")
            upsampled_edges = np.vstack([parents, np.arange(len(upsampled_vertices))]).T[1:]
            if (parents[1:] == -9999).any():
                print("Warning: some nodes have no parent, this is likely due to a disconnected graph. SKIPPING")
                continue
            try:
                upsampled_vertices, upsampled_edges = filter_duplicate_vertices(upsampled_vertices, upsampled_edges)
            except:
                import ipdb;ipdb.set_trace()  # fmt: skip
            if debug_timing:
                tim.tocTic("skeleton filter dups")
            
            highres_ls = o3d.geometry.LineSet()
            highres_ls.points = o3d.utility.Vector3dVector(upsampled_vertices)
            highres_ls.lines = o3d.utility.Vector2iVector(upsampled_edges)
            highres_ls.paint_uniform_color([0, 0, 1])
            
            vertex_pcd = o3d.geometry.PointCloud()
            vertex_pcd.points = o3d.utility.Vector3dVector(upsampled_vertices)
            vertex_pcd.paint_uniform_color([0, 1, 0])
            
            downsampled_vertices, _, downsampled_edges = downsample_ls(upsampled_vertices, upsampled_edges, 3000)
            down_ls = o3d.geometry.LineSet()
            down_ls.points = o3d.utility.Vector3dVector(downsampled_vertices)
            down_ls.lines = o3d.utility.Vector2iVector(downsampled_edges)
            down_ls.paint_uniform_color([1, 0, 0])
            down_vertex_pcd = o3d.geometry.PointCloud()
            down_vertex_pcd.points = o3d.utility.Vector3dVector(downsampled_vertices)
            down_vertex_pcd.paint_uniform_color([1, 0, 0])
            # o3d.visualization.draw_geometries([highres_ls, down_ls, down_vertex_pcd])

                
            data_item = {"scan_points": filtered_source_pcd.point.positions.numpy(), 
                         "scan_point_classes": np.ones(len(filtered_source_pcd.point.positions.numpy())), 
                         "scan_point_colors": filtered_source_pcd.point.colors.numpy(),
                         "skeleton_vertices": upsampled_vertices,
                         "skeleton_edges": upsampled_edges,
                         }

            torch.save(data_item, output_file)
            if split == "val" or split == "test":
                o3d.t.io.write_point_cloud(output_file.replace(".pt", ".xyz").replace(output_file.split('/')[-2], output_file.split('/')[-2]+"_pcd"), filtered_source_pcd)

            

if __name__ == "__main__":
    main()
