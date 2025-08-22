import os

import fpsample
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from DiffTS.tree_utils.file import load_data_npz
from DiffTS.tree_utils.tree import prune_skeleton, repair_skeleton
from DiffTS.utils.graph_utils import (downsample_skeleton_legacy,
                                                 upsample_skeleton)
from DiffTS.utils.pcd_transforms import *

class SyntheticTreesDataset(Dataset):
    def __init__(self, data_dir, split, cfg):
        super().__init__()
        self.debug = False
        self.cfg = cfg
        self.data_dir = os.path.join(data_dir, split)
        
        self.preload_data = cfg['preload']
        self.scan_prune_dist = cfg['scan_prune_dist']
        self.resolution = cfg['cond_resolution']
        self.node_resolution = cfg['node_resolution']
        self.num_nodes = cfg['num_nodes']
        self.num_condition_pts = cfg['num_points']
        self.prune_min_radius = cfg['prune_min_radius']
        self.prune_length = cfg['prune_length']
        self.node_count_vox_size = cfg['node_count_vox_size']
        self.scan_farthps = cfg['scan_farthps']
        self.varieties = cfg['varieties']
        self.debug_vis = cfg['debug_vis']
        self.dataset_norm = cfg['dataset_norm']
        self.multiply_data = cfg['multiply_data']
        self.overfit = cfg['overfit']

        self.split = split
        self.cache_maps = {}
        
        self.points_datapath = os.listdir(self.data_dir)
        self.points_datapath = [os.path.join(self.data_dir, p) for p in self.points_datapath if not ('pine' in p or 'ginkgo' in p or 'walnut' in p)]
        
        self.data_stats = {'mean': None, 'std': None}

        self.nr_data = len(self.points_datapath)
        
        if self.preload_data:
            self.cached_data = self.preload_data_fn(self.points_datapath, self.num_condition_pts)

        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
        return np.squeeze(points, axis=0)
    
    def preload_data_fn(self, points_datapath, num_condition_pts):
        cached_data = []
        print('Preloading data...')
        for index in tqdm(range(len(points_datapath))):
            scan_points, gt_skeleton, highres_gt_skeleton_nodes, highres_gt_skeleton_edges, gt_branch_id_mask, gt_branch_glob_ids, num_sample_points, scan_point_classes, scan_point_colors = SyntheticTreesDataset.process_data(points_datapath[index], prune_min_radius=self.prune_min_radius, prune_length=self.prune_length, scan_prune_dist=self.scan_prune_dist, node_count_vox_size=self.node_count_vox_size)
            
            down_masks = []
            cached_data.append([scan_points, gt_skeleton, highres_gt_skeleton_nodes, highres_gt_skeleton_edges, gt_branch_id_mask, gt_branch_glob_ids, num_sample_points, scan_point_classes, scan_point_colors, down_masks])
        return cached_data
    
    @staticmethod
    def process_data(datapath, sample_graph=True, sample_dist=0.005, node_count_vox_size=0.15, prune_min_radius=0.002, prune_length=0.05, scan_prune_dist=0.0):
        # print("prune settings", prune_min_radius, prune_length)
        scan_points_st, skeleton = load_data_npz(datapath)
        scan_points = scan_points_st.xyz
        scan_point_classes = scan_points_st.class_l
        scan_point_colors = scan_points_st.rgb.astype(np.float32)
        if prune_min_radius > 0.0 or prune_length > 0.0:
            print("Pruning skeleton")
            skeleton = prune_skeleton(skeleton, min_radius_threshold=prune_min_radius, length_threshold=prune_length)
        skeleton = repair_skeleton(skeleton)
        
        if sample_graph:
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(scan_points)
            scan_pcd = scan_pcd.voxel_down_sample(voxel_size=node_count_vox_size)
            num_sample_points = len(scan_pcd.points)
            highres_skeleton_nodes, highres_skeleton_edges, branch_id_mask, branch_glob_ids = upsample_skeleton(skeleton, sample_dist)
        else:
            raise NotImplementedError
            
        # prune scan points that are far from the skeleton
        if scan_prune_dist > 0.0:
            mesh = skeleton.to_o3d_tubes()
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            # Compute the signed distance for N random points
            dists = np.abs(scene.compute_signed_distance(scan_points.astype(np.float32)).numpy())
            prune_mask = dists < scan_prune_dist
            scan_points = scan_points[prune_mask]
            scan_point_classes = scan_point_classes[prune_mask]
            scan_point_colors = scan_point_colors[prune_mask]
        return scan_points.astype(np.float32), skeleton, highres_skeleton_nodes, highres_skeleton_edges, branch_id_mask, branch_glob_ids, num_sample_points, scan_point_classes, scan_point_colors 
    
    def vis_batch(self, skeleton_points, highres_skeleton_nodes, scan_points, gt_skeleton):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scan_points)
        pcd.paint_uniform_color([0,0,1])
        skel_pcd = o3d.geometry.PointCloud()
        skel_pcd.points = o3d.utility.Vector3dVector(skeleton_points)
        skel_pcd.paint_uniform_color([1,0,0])
        highres_skel_pcd = o3d.geometry.PointCloud()
        highres_skel_pcd.points = o3d.utility.Vector3dVector(highres_skeleton_nodes)
        highres_skel_pcd.paint_uniform_color([0,1,0])
        gt_skel_tubes = gt_skeleton.to_o3d_tubes()
        o3d.visualization.draw_geometries([pcd, highres_skel_pcd, skel_pcd, gt_skel_tubes])
        
    def __getitem__(self, index):
        index = index % self.nr_data
        if self.preload_data:
            scan_points, gt_skeleton, highres_gt_skeleton_nodes, highres_gt_skeleton_edges, gt_branch_id_mask, gt_branch_glob_ids, num_sample_points, scan_point_classes, scan_point_colors, down_masks = self.cached_data[index]
        else:
            scan_points, gt_skeleton, highres_gt_skeleton_nodes, highres_gt_skeleton_edges, gt_branch_id_mask, gt_branch_glob_ids, num_sample_points, scan_point_classes, scan_point_colors = self.process_data(datapath=self.points_datapath[index], prune_min_radius=self.prune_min_radius, prune_length=self.prune_length, scan_prune_dist=self.scan_prune_dist, node_count_vox_size=self.node_count_vox_size)
        if self.num_condition_pts == "auto":
            # voxel downsampling
            # shuffle the points
            rand_perm = torch.randperm(scan_points.shape[0])
            scan_points = scan_points[rand_perm]
            scan_point_classes = scan_point_classes[rand_perm]
            scan_point_colors = scan_point_colors[rand_perm]
            
            _, mapping = ME.utils.sparse_quantize(coordinates=scan_points / self.resolution, return_index=True, device="cpu")
            scan_points = scan_points[mapping]
            scan_point_classes = scan_point_classes[mapping]
            scan_point_colors = scan_point_colors[mapping]
        else:
            samples_idx = fpsample.bucket_fps_kdline_sampling(scan_points, self.num_condition_pts, h=9)
            scan_points = scan_points[samples_idx]
            scan_point_classes = scan_point_classes[samples_idx]
            scan_point_colors = scan_point_colors[samples_idx]
        
        if self.num_nodes == "auto":
            num_skel_nodes = num_sample_points
        else:
            num_skel_nodes = self.num_nodes
        
        skeleton_points, node_parent_ids = downsample_skeleton_legacy(gt_skeleton, highres_gt_skeleton_nodes, highres_gt_skeleton_edges, gt_branch_id_mask, gt_branch_glob_ids, num_sample_points=num_skel_nodes)
        highres_node_parent_ids = highres_gt_skeleton_edges[:, 0]
        
        if self.split == 'train':
            p_concat = np.concatenate((skeleton_points, scan_points, highres_gt_skeleton_nodes), axis=0)
            p_concat = self.transforms(p_concat)
            skeleton_points = torch.tensor(p_concat[:len(skeleton_points)], dtype=torch.float32)
            scan_points = torch.tensor(p_concat[len(skeleton_points):len(skeleton_points)+len(scan_points)], dtype=torch.float32)
            highres_gt_skeleton_nodes = torch.tensor(p_concat[len(skeleton_points)+len(scan_points):], dtype=torch.float32)
        else:
            skeleton_points = torch.tensor(skeleton_points, dtype=torch.float32)
            scan_points = torch.tensor(scan_points, dtype=torch.float32)
            highres_gt_skeleton_nodes = torch.tensor(highres_gt_skeleton_nodes, dtype=torch.float32)

        if self.debug:
            print("tree", self.points_datapath[index], "skeleton_points", skeleton_points.shape, "scan_points", scan_points.shape)  
            out_pcd = o3d.t.geometry.PointCloud()
            out_pcd.point['positions'] = o3d.core.Tensor(scan_points, dtype=o3d.core.Dtype.Float32)
            out_pcd.paint_uniform_color([0,0,1.0])
            out_pcd.point['colors'] = o3d.core.Tensor(scan_point_colors)
            o3d.io.write_point_cloud(os.path.join(os.path.dirname(self.points_datapath[index]),"scan_pcd.ply"), out_pcd.to_legacy())
            out_pcd2 = o3d.t.geometry.PointCloud()
            out_pcd2.point['positions'] = o3d.core.Tensor(skeleton_points.numpy(), dtype=o3d.core.Dtype.Float32)
            out_pcd2.point['node_parent_ids'] = o3d.core.Tensor(node_parent_ids[:, None].numpy(), dtype=o3d.core.Dtype.UInt8)
            o3d.t.io.write_point_cloud(os.path.join(os.path.dirname(self.points_datapath[index]),"skeleton_pcd.ply"), out_pcd2)
            self.vis_batch(skeleton_points, highres_gt_skeleton_nodes, scan_points, gt_skeleton)

        normalization_factors = (np.array((0,0,0)), 1)
        
        return  (skeleton_points,
            torch.tensor(highres_gt_skeleton_nodes, dtype=torch.float32),
            torch.tensor(node_parent_ids),
            torch.tensor(scan_points, dtype=torch.float32),
            torch.tensor(scan_point_classes),
            torch.tensor(scan_point_colors, dtype=torch.float32),
            self.points_datapath[index],
            torch.tensor(highres_node_parent_ids),
            normalization_factors,
            )

    def __len__(self):
        return self.nr_data * self.multiply_data if self.split == 'train' else self.nr_data
