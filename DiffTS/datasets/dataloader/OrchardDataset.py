import os

import fpsample
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from DiffTS.utils.pcd_transforms import *
from DiffTS.utils.utils import vis_flow_vecs


class OrchardDataset(Dataset):
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
        self.node_fps = cfg['node_fps']
        self.varieties = cfg['varieties']
        self.debug_vis = cfg['debug_vis']
        self.dataset_norm = cfg['dataset_norm']
        self.multiply_data = cfg['multiply_data']
        self.overfit = cfg['overfit']

        self.split = split
        self.cache_maps = {}
        self.points_datapath = [os.path.join(self.data_dir, p) for p in os.listdir(self.data_dir)]
        
        # compute normalization stats
        if self.dataset_norm:
            if not os.path.exists(os.path.join(data_dir, 'variety_max_xy_extent.npy')):
                self.variety_max_xy_extent = self.compute_variety_max_xy_extent()
                np.save(os.path.join(data_dir, 'variety_max_xy_extent.npy'), self.variety_max_xy_extent)
            else:
                self.variety_max_xy_extent = np.load(os.path.join(data_dir, 'variety_max_xy_extent.npy'), allow_pickle=True).item()
                print('Loaded data stats')

        self.nr_data = len(self.points_datapath)
        print('The size of %s data is %d'%(self.split,self.nr_data))
    
    def compute_variety_max_xy_extent(self):
        # compute extents for x and y for each variety
        variety_max_xy_extent = {}
        print("Computing variety max xy extent", self.points_datapath[0])
        for tree in tqdm(self.points_datapath):
            raw_data = torch.load(tree)
            scan_points = raw_data['scan_points']
            scan_points -= np.mean(scan_points, axis=0)
            variety = [v for v in self.varieties if v in tree][0]
            if variety not in variety_max_xy_extent:
                variety_max_xy_extent[variety] = 0
            max_xy_extent = np.max(np.abs(scan_points[:,:2])) 
            if max_xy_extent > variety_max_xy_extent[variety]:
                variety_max_xy_extent[variety] = max_xy_extent
        return variety_max_xy_extent

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])
        #TODO add elastic deformations

        return np.squeeze(points, axis=0)
    
    def vis_batch(self, skeleton_points, highres_skeleton_nodes, scan_points, parent_ids, node_parent_ids, scan_point_classes):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scan_points)
        pcd.paint_uniform_color([0,0,1])
        skel_pcd = o3d.geometry.PointCloud()
        skel_pcd.points = o3d.utility.Vector3dVector(skeleton_points)
        skel_pcd.paint_uniform_color([1,0,0])
        highres_skel_pcd = o3d.geometry.PointCloud()
        highres_skel_pcd.points = o3d.utility.Vector3dVector(highres_skeleton_nodes)
        highres_skel_pcd.paint_uniform_color([0,1,0])
        # gt_skel_tubes = gt_skeleton.to_o3d_tubes()
        edges = o3d.geometry.LineSet()
        edges.points = o3d.utility.Vector3dVector(skeleton_points)
        edges.lines = o3d.utility.Vector2iVector(np.stack([np.arange(skeleton_points.shape[0]), node_parent_ids], axis=1))
        
        flow_vecs = vis_flow_vecs(skeleton_points, skeleton_points[node_parent_ids], len=0.01)
        # paint parent nodes yellow
        # np.asarray(skel_pcd.colors)[np.asarray(edges.lines)[:,1]] = [1,1,0]
        np.asarray(pcd.colors)[scan_point_classes.astype(bool)] = [1,0,1]
        print("min max scan points", scan_points.min(axis=0), scan_points.max(axis=0))
        print("mean", scan_points.mean(axis=0))
        origin_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, highres_skel_pcd, skel_pcd, edges, origin_vis])    
        o3d.visualization.draw_geometries([skel_pcd, flow_vecs])  
    
    def center_and_normalize_data(self, data, filename, normalize=False):
        # move cloud xy around origin, z to 0
        cloud_offset = np.mean(data['scan_points'], axis=0).astype(np.float32)
        cloud_offset[2] = np.min(data['scan_points'][:,2]).astype(np.float32)
        
        if normalize:
            max_val = self.variety_max_xy_extent[[v for v in self.varieties if v in filename][0]]
        else:
            max_val = 1.0
        data['scan_points'] = (data['scan_points'] - cloud_offset) / max_val
        # for branch_id in data['skeleton'].branches.keys():
        #     data['skeleton'].branches[branch_id].xyz = (data['skeleton'].branches[branch_id].xyz - cloud_offset) / max_val
        data['skeleton_vertices'] = (data['skeleton_vertices'] - cloud_offset) / max_val
        
        return data, (cloud_offset, max_val)
    
    def precompute_sampled_ancestors(self, parent, is_sampled):
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
    
    def downsample_ls(self, vertices, edges, n_samples, fps=False):
        parents = np.concatenate((np.array((-1,)),edges[:, 0]))
        if fps:
            # farthest point sampling
            if n_samples > vertices.shape[0]:
                n_samples = vertices.shape[0]
            sampled_idx = fpsample.bucket_fps_kdline_sampling(vertices, n_samples, h=9)
            is_sampled = np.zeros(len(vertices), dtype=bool)
            is_sampled[sampled_idx] = True
        else:
            is_sampled = np.zeros(len(vertices), dtype=bool)
            is_sampled[:n_samples] = True
            np.random.shuffle(is_sampled)
        is_sampled[0] = True
        nearest = self.precompute_sampled_ancestors(parents, is_sampled)
        
        edges = np.stack((nearest[is_sampled], np.arange(len(vertices))[is_sampled]), axis=1)
        edges = edges[edges[:, 0] != -1]
        unique_verts, mapping = np.unique(edges.reshape(-1), return_inverse=True)
        vertices = vertices[unique_verts]
        upsampled_edges = mapping.reshape(-1, 2)
        parents = np.concatenate((np.array((-1,)), upsampled_edges[:, 0]))
        return vertices, parents, upsampled_edges
        
    def __getitem__(self, index):
        if self.overfit:
            index = 0
        else:
            index = index % self.nr_data
        # print(self.points_datapath[index])
        raw_data = torch.load(self.points_datapath[index])
        raw_data, normalization_factors = self.center_and_normalize_data(raw_data, filename=os.path.basename(self.points_datapath[index]), normalize=self.dataset_norm)
        scan_points = raw_data['scan_points']
        highres_gt_skeleton_nodes = raw_data['skeleton_vertices']
        skeleton_edges = raw_data['skeleton_edges']
        
        if self.split == 'val_sem':
            scan_point_classes = raw_data['pred_sem']
        else:
            scan_point_classes = raw_data['scan_point_classes']
        scan_point_colors = raw_data['scan_point_colors']
        
        if self.num_condition_pts == "auto":
            # voxel downsampling
            # shuffle the points
            rand_perm = torch.randperm(scan_points.shape[0])
            scan_points = scan_points[rand_perm]
            scan_point_classes = scan_point_classes[rand_perm]
            scan_point_colors = scan_point_colors[rand_perm]
            
            _, mapping = ME.utils.sparse_quantize(coordinates=scan_points / self.scan_resolution, return_index=True, device="cpu")
            scan_points = scan_points[mapping]
            scan_point_classes = scan_point_classes[mapping]
            scan_point_colors = scan_point_colors[mapping]
        else:
            if self.scan_farthps:
                if self.num_condition_pts > scan_points.shape[0]:
                    samples_idx = np.random.choice(scan_points.shape[0], self.num_condition_pts, replace=True)
                else:
                    samples_idx = fpsample.bucket_fps_kdline_sampling(scan_points, self.num_condition_pts, h=9)
            else:
                samples_idx = np.random.choice(scan_points.shape[0], self.num_condition_pts, replace=True)
            scan_points = scan_points[samples_idx]
            scan_point_classes = scan_point_classes[samples_idx]
            scan_point_colors = scan_point_colors[samples_idx]
        
        if self.num_nodes == "auto":
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(scan_points)
            # voxel downsample the scan points
            scan_pcd = scan_pcd.voxel_down_sample(voxel_size=self.node_count_vox_size)
            num_skel_nodes = len(scan_pcd.points)
        else:
            num_skel_nodes = self.num_nodes
         
        if self.split == 'train':
            skel_augm_variation = num_skel_nodes*0.15
            num_skel_nodes += np.random.randint(-skel_augm_variation, skel_augm_variation)
            
        if num_skel_nodes > len(highres_gt_skeleton_nodes):
            num_skel_nodes = len(highres_gt_skeleton_nodes)
        
        skeleton_points, node_parent_ids, _ = self.downsample_ls(highres_gt_skeleton_nodes, skeleton_edges, num_skel_nodes, fps=self.node_fps)
        highres_node_parent_ids = np.concatenate((np.array((-1,)), skeleton_edges[:, 0]))
         
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
            
        if self.debug_vis:
            print("tree", self.points_datapath[index], "skeleton_points", skeleton_points.shape, "scan_points", scan_points.shape, "orig_points", raw_data['scan_points'].shape,"highres_gt_skeleton_nodes", highres_gt_skeleton_nodes.shape)  
            self.vis_batch(skeleton_points, highres_gt_skeleton_nodes, scan_points, highres_node_parent_ids, node_parent_ids, scan_point_classes)

        return  (skeleton_points,
            highres_gt_skeleton_nodes,
            torch.tensor(node_parent_ids),
            scan_points,
            torch.tensor(scan_point_classes),
            torch.tensor(scan_point_colors),
            self.points_datapath[index],
            torch.tensor(highres_node_parent_ids),
            normalization_factors,
            )

    def __len__(self):
        if self.overfit:
            return 20 if self.split == 'train' else 1
        else:
            return self.nr_data * self.multiply_data if self.split == 'train' else self.nr_data

##################################################################################################
