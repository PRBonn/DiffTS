import glob
import os
import pickle
import subprocess
import tempfile
from collections import defaultdict

import click
import networkx as nx
import numpy as np
import open3d as o3d
import scipy.io as sio
import torch
from networkx import to_scipy_sparse_array
from plyfile import PlyData
from pygsp import graphs, reduction
from pykeops.torch import LazyTensor
from scipy.sparse import csgraph, csr_matrix, find
from tqdm import tqdm

from DiffTS.tree_utils.file import load_data_npz
from DiffTS.tree_utils.operations import sample_o3d_lineset
from DiffTS.tree_utils.tree import prune_skeleton
from DiffTS.utils.metrics import ChamferDistance, PrecisionRecall
from DiffTS.utils.pytimer import Timer

ted_path = "/home/elias/tools/tree-similarity/build/ted"


def center_and_normalize_data(data, full_pcd, filename, variety_max_xy_extent):
    # move cloud xy around origin, z to 0
    varieties = ['Agarwood', 'ajianglanren', 'fenghuangmu', 'FloodedGum', 'Lemon', 'LombardyPoplar', 'shanshu', 'Tibetan_Cherry', 'xiaoyelanren', 'zhangshu']
    cloud_offset = np.mean(full_pcd, axis=0).astype(np.float32)
    cloud_offset[2] = np.min(full_pcd[:,2]).astype(np.float32)
    max_val = variety_max_xy_extent[[v for v in varieties if v in filename][0]]
    data = (data - cloud_offset) / max_val
    return data

@click.command()
@click.option('--preds_folder', '-p', type=str, help='Path to the folder containing pt files', required=True)
@click.option('--reference_folder', '-r', type=str, help='Path to the folder containing pt files', required=True)
@click.option('--pred_format', '-f', type=str, help='Format of the input files', default="lineset")
@click.option('--filtering', '-fl', type=bool, help='Filter the skeleton', default=False)
@click.option('--prune_min_radius', '-pr', type=float, help='Minimum radius for pruning', default=0.0)
@click.option('--prune_min_length', '-pl', type=float, help='Length threshold for pruning', default=0.1)
@click.option('--normalize', '-n', type=bool, help='Normalize the data and predictions', default=True)
@click.option('--dataset', '-d', type=str, help='Dataset name')
@click.option('--sampled_eval', '-s', type=bool, help='Sampled evaluation', default=False)
@click.option('--viz', '-s', type=bool, help='Sampled evaluation', default=False)
@click.option('--compute_edit_distance', '-s', type=bool, help='Sampled evaluation', default=False)
def evaluate(preds_folder, reference_folder, pred_format, filtering, prune_min_radius, prune_min_length, normalize, dataset, sampled_eval, viz, compute_edit_distance):
    varieties = ['Agarwood', 'ajianglanren', 'fenghuangmu', 'FloodedGum', 'Lemon', 'LombardyPoplar', 'shanshu', 'Tibetan_Cherry', 'xiaoyelanren', 'zhangshu']

    skeleton_cd = ChamferDistance()
    skeleton_cd_variety = {v: ChamferDistance() for v in varieties}

    if dataset == "treenet3d" or dataset == "orchard":
        pred_type = "_0_pred_pp_edges.ply"
    else:
        pred_type = "_0_pred_nn_edges.ply"
    
    if dataset == "treenet3d" and not normalize:
        sampling_resolution = 0.01
    else:
        sampling_resolution = 0.001

    skeleton_pr = PrecisionRecall(0.0, 0.05, 100)
    variety_pr = {v: PrecisionRecall(0.0, 0.05, 100) for v in varieties}
    if normalize:
        variety_max_xy_extent = np.load(os.path.join('/'.join(reference_folder.split('/')[:-1]), 'variety_max_xy_extent.npy'), allow_pickle=True).item()

    if pred_format == "treeqsm":
        # load treeqsm
        qsm_data = sio.loadmat(preds_folder)
        file_list = range(len(qsm_data["QSMs"][0]))
    elif pred_format == "ours":
        file_list = sorted(os.listdir(preds_folder))
        file_list = [f for f in file_list if pred_type in f]
        file_list = [f.replace(pred_type, "") + "_skeleton.ply" for f in file_list]
        if sampled_eval:
            file_list = file_list[::10]
    else:
        file_list = sorted(os.listdir(preds_folder))
        file_list = [f for f in file_list if "skeleton" in f]
        if sampled_eval:
            file_list = file_list[::10]
    tim = Timer()
    tim.tic()
    for file_name in tqdm(file_list):       
        print("Processing file:", file_name) 
        # load predictions
        if pred_format == "treeqsm":
            tree_qsm_result = qsm_data["QSMs"][0][file_name]
            tree_name = tree_qsm_result[3][0][0][0][0][0][16][0]
            file_name = tree_name + "_skeleton.ply"
            qsm_cylinders = tree_qsm_result[0][0][0]
            # radius (m)	length (m)	start_point	axis_direction	parent	extension	branch	branch_order	position_in_branch	mad	SurfCov	added	UnmodRadius (m)
            cyl_len = qsm_cylinders[1]
            cyl_start = qsm_cylinders[2]
            cyl_axis = qsm_cylinders[3]
            # generate o3d lineset from cylinders
            nodes = []
            edges = []
            for i in range(len(cyl_start)):
                # generate line from start to end
                end = cyl_start[i] + cyl_axis[i] * cyl_len[i]
                nodes.append(cyl_start[i])
                nodes.append(end)
                edges.append([i*2, i*2+1])
            nodes = np.array(nodes)
            edges = np.array(edges)
            pred_tree_skeleton = o3d.geometry.LineSet()
            pred_tree_skeleton.points = o3d.utility.Vector3dVector(nodes)
            pred_tree_skeleton.lines = o3d.utility.Vector2iVector(edges)
            
        elif pred_format == "lineset":
            pred_tree_skeleton = o3d.io.read_line_set(os.path.join(preds_folder, file_name))
            
        elif pred_format == "smarttree":
            pred_tree_skeleton = o3d.io.read_line_set(os.path.join(preds_folder, file_name))
            reference_file = os.path.join(reference_folder, file_name.replace("_skeleton.ply", ".pt"))
            reference_file_name = file_name.replace("_skeleton.ply", "")
        elif pred_format == "adtree":
            ply = PlyData.read(os.path.join(preds_folder, file_name))
            vertices = np.vstack((ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z'])).T
            edges = np.vstack(ply['edge']['vertex_indices'])
            
            pred_tree_skeleton = o3d.geometry.LineSet()
            pred_tree_skeleton.points = o3d.utility.Vector3dVector(vertices)
            pred_tree_skeleton.lines = o3d.utility.Vector2iVector(edges)
            reference_file_name = file_name.replace("_skeleton.ply", "")
            
        elif pred_format == "pcskeletor":
            topology_name = "04_skeleton_graph_*.gpickle"
            pred_nodes = o3d.io.read_point_cloud(glob.glob(os.path.join(preds_folder, file_name, "02_skeleton_*.ply"))[0])
            with open(glob.glob(os.path.join(preds_folder, file_name, topology_name))[0], 'rb') as f:
                G = pickle.load(f)
            graph_edges = np.array(G.edges)

            pred_tree_skeleton = o3d.geometry.LineSet()
            pred_tree_skeleton.points = pred_nodes.points
            pred_tree_skeleton.lines = o3d.utility.Vector2iVector(graph_edges)
            reference_file_name = file_name.replace("_skeleton", "")
        
        elif pred_format == "ours":
            pred_tree_skeleton = o3d.io.read_line_set(os.path.join(preds_folder, file_name.replace("_skeleton.ply", "") + pred_type))
            reference_file_name = file_name.replace("_skeleton.ply", "")
            
        tim.tocTic("loaded predictions")
        
        # load reference skeleton
        if dataset == "synthetic_trees":
            raw_scan_points, raw_skeleton = load_data_npz(os.path.join(reference_folder, reference_file_name+ ".npz"))
            scan_points = np.asarray(raw_scan_points.to_o3d_cloud().points)
            skeleton = raw_skeleton
            if filtering:
                skeleton = prune_skeleton(skeleton, min_radius_threshold=prune_min_radius, length_threshold=prune_min_length)
            reference_skeleton_ls = skeleton.to_o3d_lineset()
        elif dataset == "orchard":
            if "_skeleton.ply" in file_name:
                reference_file_name = os.path.join(reference_folder, file_name.replace("_skeleton.ply", ".pt"))
            else:
                reference_file_name = os.path.join(reference_folder, file_name.replace("_skeleton", ".pt"))
            
            if not os.path.exists(reference_file_name):
                print(f"Reference file {reference_file_name} does not exist, skipping")
                o3d.visualization.draw_geometries([pred_tree_skeleton])
                import ipdb;ipdb.set_trace()  # fmt: skip
                continue
            raw_data = torch.load(reference_file_name)
            scan_points = raw_data['scan_points']
            skeleton_vertices = raw_data['skeleton_vertices']
            skeleton_edges = raw_data['skeleton_edges']
            reference_skeleton_ls = o3d.geometry.LineSet()
            reference_skeleton_ls.points = o3d.utility.Vector3dVector(skeleton_vertices)
            reference_skeleton_ls.lines = o3d.utility.Vector2iVector(skeleton_edges)
        elif dataset == "treenet3d":
            reference_file_name = os.path.join(reference_folder, reference_file_name+ ".pt")
            raw_data = torch.load(reference_file_name)
            scan_points = raw_data['scan_points']
            skeleton = raw_data['skeleton']
            reference_skeleton_ls = skeleton.to_o3d_lineset()
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented yet")

        tim.tocTic("loaded reference skeleton")
        
        if normalize:  
            gt_nodes = np.asarray(reference_skeleton_ls.points)
            gt_nodes = center_and_normalize_data(gt_nodes, scan_points, file_name, variety_max_xy_extent)
            reference_skeleton_ls.points = o3d.utility.Vector3dVector(gt_nodes)
            
            pred_nodes = np.asarray(pred_tree_skeleton.points)
            pred_nodes = center_and_normalize_data(pred_nodes, scan_points, file_name, variety_max_xy_extent)
            pred_tree_skeleton.points = o3d.utility.Vector3dVector(pred_nodes)
            
        tim.tocTic("normalized data")

        gt_samples = sample_o3d_lineset(reference_skeleton_ls, sampling_resolution)
        
        tim.tocTic("sampled ground truth skeleton")
        
        if np.asarray(pred_tree_skeleton.points).shape[0] > 1e6:
            pred_skeleton_samples = np.asarray(pred_tree_skeleton.points)
        else:
            pred_skeleton_samples = sample_o3d_lineset(pred_tree_skeleton, sampling_resolution)
        
        tim.tocTic("sampled predicted skeleton")
        
        if normalize:
            pred_skeleton_samples = pred_skeleton_samples[np.all(np.abs(pred_skeleton_samples[:,:2]) < 1, axis=1)]
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_samples)
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_skeleton_samples)
        if viz:
            viz_gt_lineset = reference_skeleton_ls.paint_uniform_color([0, 1, 0])
            pred_tree_skeleton.paint_uniform_color([1, 0, 0])
            
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(scan_points)
            scan_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(viz_gt_lineset)
            vis.add_geometry(pred_tree_skeleton)
            vis.add_geometry(scan_pcd)
            
            vis.get_render_option().line_width = 20
            vis.get_render_option().point_size = 1
            vis.run()

            # o3d.visualization.draw_geometries([pred_tree_skeleton.paint_uniform_color([1, 0, 0])])
            # import ipdb;ipdb.set_trace()  # fmt: skip
        skeleton_cd.update(gt_pcd, pred_pcd)
        for v in varieties:
            if v in file_name:
                skeleton_cd_variety[v].update(gt_pcd, pred_pcd)
        skeleton_pr.update(gt_pcd, pred_pcd)
        for v in varieties:
            if v in file_name:
                variety_pr[v].update(gt_pcd, pred_pcd)
        
        tim.tocTic("computed metrics")
        
        # print("intermediate results cd", skeleton_cd.compute()[0], file_name)
        # print("cd var", {v: cd.compute()[0] for v, cd in skeleton_cd_variety.items()})
    
    # skeleton_cd.compute()
    cd_comb, cd_pred2gt, cd_gt2pred = skeleton_cd.compute()
    print(f"Chamfer Distance: ", cd_comb, cd_pred2gt, cd_gt2pred)
    pr, re, f1 = skeleton_pr.compute_auc()
    print(f"Precision: {pr}, Recall: {re}, F1: {f1}")
    variety_cd = {v: cd.compute()[0] for v, cd in skeleton_cd_variety.items()}
    
    print(f"Chamfer Distance per variety: ", variety_cd)
    variety_pr_auc = {v: pr.compute_auc() for v, pr in variety_pr.items()}
    print(f"Precision, Recall, F1 per variety: ", variety_pr_auc)
    
    # print latex code for table
    print("Latex table entry:")
    print("& {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(cd_comb*100, pr, re, f1))
    
    # print latext code for per variety chamfer distance table with one column per variety
    print("Latex table entry per variety:")
    print("& " + " & ".join(["{:.2f}".format(cd*100) for cd in variety_cd.values()]) + " \\\\")
    
    print("latex table entry per variety f1:")
    print("& " + " & ".join(["{:.2f}".format(f1) for _, (_, _, f1) in variety_pr_auc.items()]) + " \\\\")

    
if __name__ == "__main__":
    evaluate()