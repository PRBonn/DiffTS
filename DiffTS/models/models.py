import os
from os import makedirs, path

import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from tqdm import tqdm

import DiffTS.models.diff_skeletonizer as minknet
from DiffTS.models.diff_skeletonizer import (MinkGlobalEnc,
                                                        MinkUnetGlobalEnc)
from DiffTS.tree_utils.operations import sample_o3d_lineset
from DiffTS.utils.metrics import ChamferDistance, PrecisionRecall
from DiffTS.utils.postprocess import min_spanning_tree
from DiffTS.utils.utils import (compute_diffusion_params,
                                           normalize_vecs, o3d_fps_sampling, vis_flow_vecs)

class DiffusionPoints(LightningModule):
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        
        self.diff_params = compute_diffusion_params(self.hparams)
        self.model, self.partial_enc = self.init_models()
        self.dpm_scheduler = self.init_schedulers()
        self.metrics = self.init_metrics()
        
        self.w_uncond = self.hparams['train']['uncond_w']

    def init_models(self):
        in_channels = 4 if self.hparams['train']['input_feats'] == "semantics" else 6 if self.hparams['train']['input_feats'] == "colors" else 3
        encoder_cls = (
            MinkUnetGlobalEnc if self.hparams['model'].get('unet_encoder') else
            MinkGlobalEnc
        )
        cond_encoder = encoder_cls(in_channels=in_channels, out_channels=self.hparams['model']['out_dim'], downscale_model=self.hparams['model']['downscale_params'], debug=self.hparams['model']['debug'], multiscale_cond=self.hparams['model']['multiscale_cond'])
        denoising_model = minknet.AttentiveMinkUNetDiff(in_channels=6, # x,y,z,flow_x,flow_y,flow_z
                                                   out_channels=6, 
                                                   voxel_size=self.hparams['data']['cond_resolution'], 
                                                   node_voxel_size=self.hparams['data']['node_resolution'],
                                                   neighb_fusion=self.hparams['train']['neighb_fusion'],
                                                   downscale_model=self.hparams['model']['downscale_params'],
                                                   debug=self.hparams['model']['debug'],
                                                   cond_dist_limit=self.hparams['train']['cond_dist_limit'],
                                                   multiscale_cond=self.hparams['model']['multiscale_cond'])
        return denoising_model, cond_encoder
        
    def init_schedulers(self):
        schedulers =  [DPMSolverMultistepScheduler(
                num_train_timesteps=self.hparams['diff']['t_steps'],
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2
            ) for _ in range(self.hparams['train']['batch_size'])]
        [scheduler_it.set_timesteps(self.diff_params["s_steps"]) for scheduler_it in schedulers]
        return schedulers
    
    def init_metrics(self):
        return {
            "node_chamfer_distance": ChamferDistance(),
            "nn_skeleton_cd": ChamferDistance(),
            "pp_skeleton_cd": ChamferDistance(),
            "precision_recall": PrecisionRecall(0, 0.05, 100),
            "chamfer_distance_variety": {v: ChamferDistance() for v in self.hparams['data']['varieties']}
        }

    def scheduler_to_cuda(self):
        for batch_id in range(self.hparams['train']['batch_size']):    
            self.dpm_scheduler[batch_id].timesteps = self.dpm_scheduler[batch_id].timesteps.cuda()
            self.dpm_scheduler[batch_id].betas = self.dpm_scheduler[batch_id].betas.cuda()
            self.dpm_scheduler[batch_id].alphas = self.dpm_scheduler[batch_id].alphas.cuda()
            self.dpm_scheduler[batch_id].alphas_cumprod = self.dpm_scheduler[batch_id].alphas_cumprod.cuda()
            self.dpm_scheduler[batch_id].alpha_t = self.dpm_scheduler[batch_id].alpha_t.cuda()
            self.dpm_scheduler[batch_id].sigma_t = self.dpm_scheduler[batch_id].sigma_t.cuda()
            self.dpm_scheduler[batch_id].lambda_t = self.dpm_scheduler[batch_id].lambda_t.cuda()
            self.dpm_scheduler[batch_id].sigmas = self.dpm_scheduler[batch_id].sigmas.cuda()
            
            # reset scheduler for new sample otherwise it will keep stored the last sample from the previous batch
            self.dpm_scheduler[batch_id].model_outputs = [None] * self.dpm_scheduler[batch_id].config.solver_order
            self.dpm_scheduler[batch_id].lower_order_nums = 0
            self.dpm_scheduler[batch_id]._step_index = None

    def q_sample(self, x, t, noise):
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        return self.diff_params["sqrt_alphas_cumprod"][t][:,None,None].cuda() * x + \
                self.diff_params["sqrt_one_minus_alphas_cumprod"][t][:,None,None].cuda() * noise

    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_cond, _ = self.forward(x_t, x_cond, t)      
        x_uncond, _ = self.forward(x_t, x_uncond, t)
        return [x_uncond_it + self.hparams['train']['uncond_w'] * (x_cond_it - x_uncond_it) for x_uncond_it, x_cond_it in zip(x_uncond, x_cond)]
    
    def p_sample_loop(self, x_init, x_t, x_cond, x_uncond):
        print('Sampling...')
        batch_size = len(x_init)
        
        # reset scheduler
        self.scheduler_to_cuda()

        for t_in in tqdm(range(len(self.dpm_scheduler[0].timesteps))):
            t = torch.ones(batch_size).cuda().long() * self.dpm_scheduler[0].timesteps[t_in].cuda()
            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            input_noise = [x_t_it - x_init_it for x_t_it, x_init_it in zip(x_t, x_init)]
            x_t = [x_init_it + self.dpm_scheduler[iter_n].step(noise_t_it, t[0], input_noise_it)['prev_sample'] for (iter_n, x_init_it), noise_t_it, input_noise_it in zip(enumerate(x_init), noise_t, input_noise)]
            torch.cuda.empty_cache()
        return x_t
    
    def getDiffusionLoss(self, x, y):
        losses = []
        for batch_it in range(len(x)):
            losses.append(F.mse_loss(x[batch_it], y[batch_it]))
            
        return torch.mean(torch.stack(losses))


    def forward(self, x_full, cond_input, t):
        part_feat = self.partial_enc(self.torch_to_mink(cond_input, resolution=self.hparams['data']['cond_resolution']))
        out = self.model(self.torch_to_mink(x_full, resolution=self.hparams['data']['node_resolution']),
                                   part_feat, t)
        torch.cuda.empty_cache()
        part_mid_feats = []
        return out, part_mid_feats

    def torch_to_mink(self, x_feats, resolution):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats[:,:4].clone()
        x_coord[:,1:] = torch.round(x_feats[:,1:4] / resolution)
        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )
        torch.cuda.empty_cache()

        return x_t
    
    def generate_cond_input(self, batch):
        # add classes as feats
        if self.hparams['train']['input_feats'] == "semantics":
            cond_feats = [torch.cat((cond_pts, pts_cls.unsqueeze(dim=-1)), dim=-1) for cond_pts, pts_cls in zip(batch['pcd_conditioning_pts'], batch['scan_point_classes'])]
        elif self.hparams['train']['input_feats'] == "colors":
            cond_feats = [torch.cat((cond_pts, pts_col), dim=-1) for cond_pts, pts_col in zip(batch['pcd_conditioning_pts'], batch['scan_point_colors'])]
        else:
            cond_feats = batch['pcd_conditioning_pts']
        return cond_feats

    def training_step(self, batch:dict, batch_idx):
        torch.cuda.empty_cache()
        # initial random noise
        noise = [torch.randn(nodes_it.shape, device=self.device) for nodes_it in batch['pcd_nodes']]
        
        # sample step t
        t = torch.randint(0, self.diff_params["t_steps"], size=(len(batch['pcd_nodes']),)).cuda()
        
        # if debugging fix t to 0
        if self.hparams['model']['debug']:
            t = t * 0 + int(self.diff_params["t_steps"]/2) - 1
        
        # create noised nodes [B, N, 3]
        node_t_sample = [nodes_it + self.q_sample(torch.zeros_like(nodes_it), t_it, noise_it) for nodes_it, noise_it, t_it in zip(batch['pcd_nodes'], noise, t)]
        # compute gt flow vectors [B, N, 3]
        rel_parent_coords = [normalize_vecs(nodes_it[parent_it] - nodes_it) for nodes_it, parent_it in zip(batch['pcd_nodes'], batch['node_parent_ids'])]
        # create noised flow vectors [B, N, 3]
        parent_noise = [torch.randn_like(parent_it, device=self.device) for parent_it in rel_parent_coords]
        parent_t_sample = [(rel_parent_coords_it) + self.q_sample(torch.zeros_like(rel_parent_coords_it), t_it, parent_noise_it) for rel_parent_coords_it, parent_noise_it, nodes_it, t_it in zip(rel_parent_coords, parent_noise, batch['pcd_nodes'], t)]

        # create network input by concatenating noised nodes and noised flows [B, N, 6]
        full_t_sample = [torch.cat((node_t_sample_it, parent_t_sample_it), dim=-1).squeeze(0) for node_t_sample_it, parent_t_sample_it in zip(node_t_sample, parent_t_sample)]

        # generate conditioning feats according to config
        cond_feats = self.generate_cond_input(batch)
            
        # for classifier-free guidance switch between conditional and unconditional training
        if torch.rand(1) > self.hparams['train']['uncond_prob'] or len(batch['pcd_nodes']) == 1:
            cond_input = cond_feats
        else:
            cond_input = [torch.zeros_like(x) for x in cond_feats]
        
        # predict noise
        denoise_t, _ = self.forward(full_t_sample, cond_input, t)
        
        # compute L2 loss 
        noise = [torch.cat((noise_it, parent_noise_it), dim=-1) for noise_it, parent_noise_it in zip(noise, parent_noise)]
        loss_mse = self.getDiffusionLoss(denoise_t, noise)
        loss_mean = torch.mean(torch.stack([denoise_it.mean() for denoise_it in denoise_t]))**2
        loss_std = torch.mean(torch.stack([(denoise_it.std() - 1.) for denoise_it in denoise_t]))**2
        loss = loss_mse + self.hparams['diff']['reg_weight'] * (loss_mean + loss_std)
        self.log('train/loss_mse', loss_mse)
        self.log('train/loss_mean', loss_mean)
        self.log('train/loss_std', loss_std)
        self.log('train/loss', loss)
        torch.cuda.empty_cache()

        return loss

    def diffusion_inference(self, batch):
        batch_size = len(batch['pcd_conditioning_pts'])
        init_pts = [o3d_fps_sampling(batch['pcd_conditioning_pts'][b], batch['pcd_nodes'][b].shape[0]) for b in range(batch_size)]  
        init_pts = [torch.cat((init_pts_it, torch.zeros_like(init_pts_it)), dim=-1) for init_pts_it in init_pts]
        
        if self.hparams['train']['qsample_inference']:
            noise = [torch.randn(init_pts_it.shape, device=self.device) for init_pts_it in init_pts]
            t = torch.full((len(batch['pcd_nodes']),), self.diff_params["t_steps"] - 1, device=self.device).long()
            noised_init_pts = [init_pts_it + self.q_sample(torch.zeros_like(init_pts_it), t_it, noise_it).squeeze(0) for init_pts_it, noise_it, t_it in zip(init_pts, noise, t)]
        else:
            noised_init_pts = [init_pts_it + torch.randn(init_pts_it.shape, device=self.device) for init_pts_it in init_pts]
        cond_input = self.generate_cond_input(batch)
        pred_pts = self.p_sample_loop(init_pts, noised_init_pts, cond_input, [torch.zeros_like(x) for x in cond_input])
        return pred_pts

    def generate_prediction(self, batch):
        with torch.no_grad():
            model_output = self.diffusion_inference(batch)

            for i in range(len(batch['pcd_nodes'])):
                node_pred = model_output[i][:,:3]
                
                parent_pred = node_pred + model_output[i][:,3:]
                
                dist_pts = node_pred.norm(dim=-1)
                valids = dist_pts < self.hparams['data']['max_range']
                if valids.sum() < len(batch['pcd_nodes'][i])//2:
                    print("Less than half of the nodes inside max range!! SKIPPING...")
                    continue
                node_pred = node_pred[valids]
                parent_pred = parent_pred[valids]
                
    def nn_skeleton(self, node_pred, parent_pred, pcd_gt, pcd_filename=None, debug=False):
        dists = torch.cdist(parent_pred, node_pred)
        dists[torch.eye(dists.shape[0]).bool()] = torch.inf
        pred_parent_ids = dists.argmin(dim=-1)
        
        # compute sampled skeleton graph chamfer distance
        nn_edges = o3d.geometry.LineSet()
        nn_edges.points = o3d.utility.Vector3dVector(node_pred.cpu().detach().numpy())
        nn_edges.lines = o3d.utility.Vector2iVector(np.stack((np.arange(node_pred.shape[0]), pred_parent_ids.cpu().detach().numpy()), axis=1))
        skeleton_points = sample_o3d_lineset(nn_edges, 0.001)
        skeleton_pcd = o3d.geometry.PointCloud()
        skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_points)
        self.metrics["nn_skeleton_cd"].update(pcd_gt, skeleton_pcd)
        if pcd_filename != None:
            o3d.io.write_line_set(f'{self.logger.log_dir}/generated_pcd/{pcd_filename.split("/")[-1].split(".")[0]}_{self.current_epoch}_pred_nn_edges.ply', nn_edges)
        if debug:
            pcd_gt.paint_uniform_color([0.,1.,0.])
            skeleton_pcd.paint_uniform_color([1.,0.,0.])
            o3d.visualization.draw_geometries([pcd_gt, skeleton_pcd], window_name='NN Skeleton')
        return skeleton_pcd, nn_edges
        
    def pp_skeleton(self, node_pred, parent_pred, pcd_gt, cond_pts, pcd_filename=None, debug=False):
        pp_edges = min_spanning_tree(node_pred.cpu().detach().float(), cond_pts.cpu().detach().float(), parent_pred.cpu().detach().float(), max_bridge_dist=self.hparams["data"]["pp_max_bridge_dist"], connected_components_filt=self.hparams["data"]["connected_components_filt"], averaging=False)
        skeleton_points_pp = sample_o3d_lineset(pp_edges, 0.001)
        skeleton_pcd_pp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(skeleton_points_pp))
        self.metrics["pp_skeleton_cd"].update(pcd_gt, skeleton_pcd_pp)
        if pcd_filename != None:
            o3d.io.write_line_set(f'{self.logger.log_dir}/generated_pcd/{pcd_filename.split("/")[-1].split(".")[0]}_{self.current_epoch}_pred_pp_edges.ply', pp_edges)
        if debug:
            pcd_gt.paint_uniform_color([0.,1.,0.])
            skeleton_pcd_pp.paint_uniform_color([1.,0.,0.])
            o3d.visualization.draw_geometries([pcd_gt, skeleton_pcd_pp, pp_edges], window_name='PP Skeleton')
        return skeleton_pcd_pp, pp_edges
    
    def variety_cd(self, pred_sampled_pcd, pcd_gt, filename):
        # update variety chamfer distances
        if len(self.hparams['data']['varieties']) > 1:
            curr_variety = [v for v in self.metrics["chamfer_distance_variety"].keys() if v in filename][0]
            if curr_variety in self.hparams['data']['varieties']:
                self.metrics["chamfer_distance_variety"][curr_variety].update(pcd_gt, pred_sampled_pcd)
            else:
                print(f'Variety {curr_variety} not in varieties list. Skipping...')
                
    def save_network_preds(self, cond_pts, i, node_pred, parent_pred, pcd_gt, nn_edges, pcd_filename):
        pcd_node_pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node_pred.cpu().detach().numpy()))
        o3d.io.write_point_cloud(f'{self.logger.log_dir}/generated_pcd/{pcd_filename.split("/")[-1].split(".")[0]}_{self.current_epoch}_pred.ply', pcd_node_pred)
        o3d.io.write_point_cloud(f'{self.logger.log_dir}/generated_pcd/{pcd_filename.split("/")[-1].split(".")[0]}_{self.current_epoch}_gt.ply', pcd_gt)
        cond_pcd = o3d.geometry.PointCloud()
        cond_pcd.points = o3d.utility.Vector3dVector(cond_pts)
        o3d.io.write_point_cloud(f'{self.logger.log_dir}/generated_pcd/{pcd_filename.split("/")[-1].split(".")[0]}_{self.current_epoch}_cond.ply', cond_pcd)
        open_edges = o3d.geometry.LineSet()
        open_parent_nodes = torch.cat((node_pred, parent_pred), dim=0)
        open_edges.points = o3d.utility.Vector3dVector(open_parent_nodes.cpu().detach().numpy())
        open_edges.lines = o3d.utility.Vector2iVector(np.stack((np.arange(node_pred.shape[0]), np.arange(node_pred.shape[0],node_pred.shape[0]+parent_pred.shape[0])), axis=1))
        o3d.io.write_line_set(f'{self.logger.log_dir}/generated_pcd/{pcd_filename.split("/")[-1].split(".")[0]}_{self.current_epoch}_pred_open_edges.ply', open_edges)
        o3d.io.write_line_set(f'{self.logger.log_dir}/generated_pcd/{pcd_filename.split("/")[-1].split(".")[0]}_{self.current_epoch}_pred_nn_edges.ply', nn_edges)
    
    def eval_and_save(self, batch, save_raw=False, post_process=False, save_pcds=False, vis_output=False):
        if save_pcds:
            os.makedirs(f'{self.logger.log_dir}/generated_pcd/', exist_ok=True)
        with torch.no_grad():
            model_output = self.diffusion_inference(batch)

            for i in range(len(batch['pcd_nodes'])):
                normalization_factors = batch['normalization_factors'][i]
                cloud_offset, max_val = normalization_factors
                cond_pts = batch['pcd_conditioning_pts'][i].cpu().detach().float() * max_val + cloud_offset
                node_pred = model_output[i][:,:3].cpu() * max_val + cloud_offset
                parent_offset = model_output[i][:,3:].cpu() * max_val
                parent_pred = node_pred + parent_offset
                
                dist_pts = node_pred.norm(dim=-1)
                valids = dist_pts < self.hparams['data']['max_range']
                if valids.sum() < len(batch['pcd_nodes'][i])//2:
                    print("Less than half of the nodes inside max range!! SKIPPING...")
                    continue
                node_pred = node_pred[valids]
                parent_pred = parent_pred[valids]
                
                pcd_pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node_pred.cpu().detach().numpy()))
                gt_nodes = batch['full_pcd_nodes'][i].cpu().detach().numpy() * max_val + cloud_offset
                pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_nodes))
                self.metrics["node_chamfer_distance"].update(pcd_gt, pcd_pred)
                
                nn_sampled_pcd, nn_edges = self.nn_skeleton(node_pred, parent_pred, pcd_gt)
                if save_pcds:
                    self.save_network_preds(cond_pts, i, node_pred, parent_pred, pcd_gt, nn_edges, batch['filename'][i])

                if post_process:
                    pp_sampled_pcd, pp_edges = self.pp_skeleton(node_pred, parent_pred, pcd_gt, cond_pts, pcd_filename=batch["filename"][i] if save_pcds else None)
                    
                
                self.variety_cd(pcd_gt, pcd_gt=pp_sampled_pcd if post_process else nn_sampled_pcd, filename=batch['filename'][i])
                self.metrics["precision_recall"].update(pcd_gt, pp_sampled_pcd if post_process else nn_sampled_pcd)
                
                if vis_output:
                    # print("file name", batch['filename'][i])
                    # skeleton_pcd_pp.paint_uniform_color([1.,0.,0.])
                    pcd_gt.paint_uniform_color([1.,1.,0.])
                    pcd_pred.paint_uniform_color([1.,0.,0.])
                    filtered_pred_nodes = o3d.geometry.PointCloud()
                    filtered_pred_nodes.points = o3d.utility.Vector3dVector(pp_edges.points)
                    filtered_pred_nodes.paint_uniform_color([1.,0.,0.])
                    np.asarray(filtered_pred_nodes.colors)[np.asarray(pp_edges.lines)[:,0]] = [0,0,1.]
                    np.asarray(filtered_pred_nodes.colors)[np.asarray(pp_edges.lines)[:,1]] = [0,0,1.]
                    cond_pcd = o3d.geometry.PointCloud()
                    cond_pcd.points = o3d.utility.Vector3dVector((batch['pcd_conditioning_pts'][i].cpu().detach().numpy() * max_val + cloud_offset))
                    cond_pcd.paint_uniform_color([0.,1.,0.])
                    flow_vecs = vis_flow_vecs(node_pred, parent_pred, len=0.02).paint_uniform_color([1,0,0])
                    o3d.visualization.draw_geometries([cond_pcd, pcd_gt, filtered_pred_nodes, pp_edges, flow_vecs])
                    
        return self.metrics["node_chamfer_distance"].compute(), self.metrics["nn_skeleton_cd"].compute(), self.metrics["pp_skeleton_cd"].compute(), self.metrics["precision_recall"].compute_auc(), {key: y.compute()[0] for key, y in self.metrics["chamfer_distance_variety"].items()}

        
    def validation_step(self, batch:dict, batch_idx):
        if self.hparams['data']['test_w_uncond']:
            self.test_w_uncond(batch)
            return
        if batch_idx != 0:
            return
        
        self.model.eval()
        self.partial_enc.eval()
        
        node_cd, nn_skeleton_cd, pp_skeleton_cd, prec_rec_f1, variety_chamfer_dists = self.eval_and_save(batch)

        self.log('val/node_chamfer_dist', node_cd[0], on_step=True)
        self.log('val/node_chamfer_dist_p2g', node_cd[1], on_step=True)
        self.log('val/node_chamfer_dist_g2p', node_cd[2], on_step=True)
        self.log('val/nn_skel_chamfer_dist', nn_skeleton_cd[0], on_step=True)
        self.log('val/nn_skel_chamfer_dist_p2g', nn_skeleton_cd[1], on_step=True)
        self.log('val/nn_skel_chamfer_dist_g2p', nn_skeleton_cd[2], on_step=True)
        self.log('val/pp_skel_chamfer_dist', pp_skeleton_cd[0], on_step=True)
        self.log('val/pp_skel_chamfer_dist_p2g', pp_skeleton_cd[1], on_step=True)
        self.log('val/pp_skel_chamfer_dist_g2p', pp_skeleton_cd[2], on_step=True)
        self.log('val/precision', prec_rec_f1[0], on_step=True)
        self.log('val/recall', prec_rec_f1[1], on_step=True)
        self.log('val/fscore', prec_rec_f1[2], on_step=True)
        torch.cuda.empty_cache()

        return {'val/node_chamfer_dist': node_cd[0], 'val/precision': prec_rec_f1[0], 'val/recall': prec_rec_f1[1], 'val/fscore': prec_rec_f1[2]}
    
    def on_validation_epoch_end(self):
        self.metrics["node_chamfer_distance"].reset()
        self.metrics["nn_skeleton_cd"].reset()
        self.metrics["precision_recall"].reset()
    
    def test_w_uncond(self, batch):
        print("Testing w_uncond")
        results = []
        for w_uncond in tqdm(range(5, 150, 5)):
            self.w_uncond = w_uncond/10
            chamf_dist, skel_chamf_dist, skel_chamf_dist_pp, pr_rec, var_metrics = self.eval_and_save(batch, post_process=True)
            results.append([w_uncond, chamf_dist, skel_chamf_dist, skel_chamf_dist_pp])
            self.metrics["node_chamfer_distance"].reset()
            self.metrics["nn_skeleton_cd"].reset()
            self.metrics["pp_skeleton_cd"].reset()
            print(f'w_uncond: {w_uncond/10}\tCD: {chamf_dist[0]}\tSkel CD: {skel_chamf_dist[0]}\tPP Skel CD: {skel_chamf_dist_pp[0]}')

        print(results)
        for res in results:
            print(f'w_uncond: {res[0]/10}\tCD: {res[1]}\tSkel CD: {res[2]}\tPP Skel CD: {res[3]}')
        import ipdb;ipdb.set_trace()  # fmt: skip
        
    def test_pred_std(self, batch):
        print("Testing pred std")
        results = []
        for pred_std in tqdm(range(50)):
            node_cd, nn_skeleton_cd, pp_skeleton_cd, prec_rec_f1, variety_chamfer_dists = self.eval_and_save(batch, post_process=True, save_pcds=True)
            
            results.append([node_cd[0], nn_skeleton_cd[0], pp_skeleton_cd[0]])
            self.metrics["node_chamfer_distance"].reset()
            self.metrics["nn_skeleton_cd"].reset()
            self.metrics["pp_skeleton_cd"].reset()
            
        print(results)
        results = np.array(results)
        import ipdb;ipdb.set_trace()        

    def test_step(self, batch:dict, batch_idx):
        self.model.eval()
        self.partial_enc.eval()
        
        # self.test_pred_std(batch)
        if "save_pcds" in self.hparams['data']:
            save_pcds = self.hparams['data']['save_pcds']
        else:
            save_pcds = True
        node_cd, nn_skeleton_cd, pp_skeleton_cd, prec_rec_f1, variety_chamfer_dists = self.eval_and_save(batch, post_process=True, save_pcds=save_pcds, vis_output=self.hparams['data']['vis_output'])
        print(f'Node CD Mean: {node_cd[0]}\tCD p2g: {node_cd[1]}\tCD g2p: {node_cd[2]} ')
        print(f'NN Skeleton CD Mean: {nn_skeleton_cd[0]}\tCD p2g: {nn_skeleton_cd[1]}\tCD g2p: {nn_skeleton_cd[2]} ')
        print(f'Postprocessd Skeleton CD Mean: {pp_skeleton_cd[0]}\tCD p2g: {pp_skeleton_cd[1]}\tCD g2p: {pp_skeleton_cd[2]} ')
        print(f'Precision: {prec_rec_f1[0]}\tRecall: {prec_rec_f1[1]}\tF-Score: {prec_rec_f1[2]}')
        print(f'Variety Chamfer Dists:', variety_chamfer_dists)

        self.log('test/node_chamfer_dist', node_cd[0], on_step=True)
        self.log('test/nn_skel_chamfer_dist', nn_skeleton_cd[0], on_step=True)
        self.log('test/pp_skel_chamfer_dist', pp_skeleton_cd[0], on_step=True)
        self.log('test/precision', prec_rec_f1[0], on_step=True)
        self.log('test/recall', prec_rec_f1[1], on_step=True)
        self.log('test/fscore', prec_rec_f1[2], on_step=True)
        for key, val in variety_chamfer_dists.items():
            self.log(f'test/chamfer_dist_{key}', val, on_step=True)
        torch.cuda.empty_cache()

        return {'test/node_chamfer_dist': node_cd[0], 'test/precision': prec_rec_f1[0], 'test/recall': prec_rec_f1[1], 'test/fscore': prec_rec_f1[2]}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))
        if self.hparams['train']['lr_decay_epochs']:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
            scheduler = {
                'scheduler': scheduler, # lr * 0.5
                'interval': 'epoch', # interval is epoch-wise
                'frequency': self.hparams['train']['lr_decay_epochs'], # after 5 epochs
            }

            return [optimizer], [scheduler]
        else:
            return optimizer