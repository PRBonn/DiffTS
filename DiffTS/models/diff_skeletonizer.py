import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from pykeops.torch import LazyTensor

from DiffTS.models import minkunet_blocks


def visualize_stages(stages):
    colors = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
    for i, (stage, color) in enumerate(zip(stages, colors)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stage.C[:, 1:].cpu().numpy() + (stage.tensor_stride[0] / 2))
        pcd.paint_uniform_color(color)
        print(f"num points at stage {i}: {stage.C.shape[0]}")
        o3d.visualization.draw_geometries([pcd])

class MinkGlobalEnc(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)
        self.downscale_model = kwargs.get('downscale_model', 1)
        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        cs = torch.tensor([32, 32, 64, 128, 256, 256, 128, 96, 96]) // self.downscale_model
        cs = [int(cr * x) for x in cs]
        self.embed_dim = cs[-1]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        self.multiscale_cond = kwargs.get('multiscale_cond', False)
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(cs[0], cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            minkunet_blocks.BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage2 = nn.Sequential(
            minkunet_blocks.BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage3 = nn.Sequential(
            minkunet_blocks.BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.stage4 = nn.Sequential(
            minkunet_blocks.BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x.sparse())
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        if self.debug:
            visualize_stages([x0, x1, x2, x3, x4])
        
        if self.multiscale_cond:
            return x1, x2, x3, x4
        else:
            return x4
    
class MinkUnetGlobalEnc(minkunet_blocks.MinkSemiUNet14):
    def __init__(self, **kwargs):
        in_channels = kwargs.get('in_channels', 3)
        out_channels = 256
        super().__init__(in_channels=in_channels, out_channels=out_channels)
    


class AttentiveMinkUNetDiff(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)
        self.n_neighbs = kwargs.get('n_neighbs', 8)
        self.downscale_model = kwargs.get('downscale_model', 1)
        self.neighb_fusion = kwargs.get('neighb_fusion', True)
        self.voxel_size = kwargs.get('voxel_size', 0.01)
        self.node_voxel_size = kwargs.get('node_voxel_size', 0.05)
        self.cond_dist_limit = kwargs.get('cond_dist_limit', 0.0)
        self.multiscale_cond = kwargs.get('multiscale_cond', False)

        cr = kwargs.get('cr', 1.0)
        in_channels = kwargs.get('in_channels', 3)
        out_channels = kwargs.get('out_channels', 3)
        
        self.cs = torch.tensor([32, 32, 64, 128, 256, 256, 128, 96, 96]) // self.downscale_model
        self.num_stages = 4
        self.transformer_layer_depth = 1
        self.cs = [int(cr * x) for x in self.cs] 
        self.embed_dim = self.cs[-1]
        self.run_up = kwargs.get('run_up', True)
        self.D = kwargs.get('D', 3)
        
        self.vis_cache = []
        
        self.pos_emb = minkunet_blocks.PositionalEncoder()
        
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(self.cs[0]),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(self.cs[0], self.cs[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(self.cs[0]),
            ME.MinkowskiReLU(inplace=True)
        )

        self.stages = nn.ModuleList()
        self.latents = nn.ModuleList()
        self.latemps = nn.ModuleList()
        self.stage_temps = nn.ModuleList()
        self.neighb_proj = nn.ModuleList()
        self.neighb_aggregator = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.tanh = nn.Tanh()

        # Create stages
        for i in range(2*self.num_stages):
            self.neighb_proj.append(nn.Linear(self.cs[4], self.cs[i]))
            if self.neighb_fusion == 'attention':
                self.neighb_aggregator.append(nn.Transformer(d_model=self.cs[i], nhead=4, num_encoder_layers=self.transformer_layer_depth, num_decoder_layers=self.transformer_layer_depth, dim_feedforward=self.cs[i], batch_first=True))
            self.latents.append(self._make_latent_stage(self.cs[i], self.cs[i]))
            self.latemps.append(self._make_latemp_stage(self.cs[i], self.cs[i], self.cs[i]))
            self.stage_temps.append(self._make_stage_temp(self.embed_dim, self.cs[i]))
            if i < self.num_stages:
                self.stages.append(self._make_stage(self.cs[i], self.cs[i + 1]))
            else:
                self.ups.append(self._make_up(self.cs[i], self.cs[i - 1], self.cs[i + 1], i - self.num_stages))
        self.last = nn.Sequential(
            nn.Linear(self.cs[8], int(self.cs[8]//2)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(int(self.cs[8]//2), out_channels),
        )

        self.weight_initialization()

    def _make_latent_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),              
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def _make_latemp_stage(self, in_channels, hidden_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )

    def _make_stage_temp(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_channels, out_channels),
        )

    def _make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            minkunet_blocks.BasicConvolutionBlock(in_channels, in_channels, ks=2, stride=2, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(in_channels, out_channels, ks=3, stride=1, dilation=1, D=self.D),
            minkunet_blocks.ResidualBlock(out_channels, out_channels, ks=3, stride=1, dilation=1, D=self.D),
        )

    def _make_up(self, in_channels, mid_channels, out_channels, idx):
        return nn.ModuleList([
            minkunet_blocks.BasicDeconvolutionBlock(in_channels, mid_channels, ks=2, stride=2, D=self.D),
            nn.Sequential(
                minkunet_blocks.ResidualBlock(mid_channels + self.cs[3-idx], mid_channels, ks=3, stride=1, dilation=1, D=self.D),
                minkunet_blocks.ResidualBlock(mid_channels, out_channels, ks=3, stride=1, dilation=1, D=self.D),
            )
        ])

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_timestep_embedding(self, timesteps):
        assert len(timesteps.shape) == 1 

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(torch.device('cuda'))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb
    
    def match_part_to_full(self, x_node, x_cond, stage, input_pcd=None):
        # Extract and convert coordinates to meter space
        full_c = x_node.C.clone().float()
        part_c = x_cond.C.clone().float()
        
        # Hash batch coordinates
        max_coord = full_c.max()
        full_c[:,0] *= max_coord * 2.
        part_c[:,0] *= max_coord * 2.
        
        # Add half voxel offset and convert to meter space
        full_c[:,1:] = (full_c[:,1:] + x_node.tensor_stride[0]/2) * self.node_voxel_size
        part_c[:,1:] = (part_c[:,1:] + x_cond.tensor_stride[0]/2) * self.voxel_size
        
        # Find K nearest neighbors using LazyTensor
        f_coord = LazyTensor(full_c[:,None,:])
        p_coord = LazyTensor(part_c[None,:,:])
        dist_fp = ((f_coord - p_coord)**2).sum(-1)
        match_feats = dist_fp.argKmin(self.n_neighbs, dim=1)
        
        # Get features and coordinates
        scan_neighb_feats = x_cond.F[match_feats]
        scan_neighb_coord = part_c[match_feats].float()
        node_coord = full_c
        node_feats = x_node.F
        
        # Project neighbor features
        scan_neighb_feats = self.neighb_proj[stage](scan_neighb_feats)
        
        # Apply neighbor fusion strategy
        if self.neighb_fusion == 'attention':
            scan_pose_emb = self.pos_emb(scan_neighb_coord[...,1:], node_feats.shape[-1])
            node_pose_emb = self.pos_emb(node_coord.unsqueeze(1)[...,1:], node_feats.shape[-1]).squeeze(1)
            out_feats = self.neighb_aggregator[stage](scan_neighb_feats+scan_pose_emb, 
                                                    (node_feats+node_pose_emb).unsqueeze(-2)).squeeze(-2)
        elif self.neighb_fusion == 'wMean':
            dists = (scan_neighb_coord[...,1:] - node_coord[...,1:].unsqueeze(1)).norm(dim=-1)
            dists = torch.clamp(dists, min=1e-6)
            weights = 1 / dists
            weights = weights / weights.sum(dim=1, keepdim=True)
            
            if self.cond_dist_limit:
                dist_limit = self.cond_dist_limit * x_cond.tensor_stride[0] * self.voxel_size
                weights = torch.where(dists < dist_limit, weights, torch.zeros_like(weights))
                # Renormalize non-zero weights
                weights_sum = weights.sum(dim=1, keepdim=True)
                inlier_mask = weights_sum.squeeze(-1) > 0
                weights[inlier_mask] = weights[inlier_mask] / weights_sum[inlier_mask]
                
            out_feats = (scan_neighb_feats * weights.unsqueeze(-1)).sum(dim=1)
        elif self.neighb_fusion == 'mean':
            out_feats = scan_neighb_feats.mean(dim=1)
        else:
            raise NotImplementedError(f'Neighb fusion "{self.neighb_fusion}" not implemented')

        # Debug visualization if needed
        if self.debug:
            self._debug_visualize(full_c, part_c, match_feats, 
                                 weights if self.neighb_fusion == 'wMean' else None,
                                 x_node, input_pcd)
                
        return out_feats
        
    def _debug_visualize(self, full_c, part_c, match_feats, weights=None, x_node=None, input_pcd=None):
            nodes_pcd = o3d.geometry.PointCloud()
            nodes_pcd.points = o3d.utility.Vector3dVector(full_c[:,1:].cpu().numpy())
            nodes_pcd.paint_uniform_color([1,0,0])
        
            cond_pcd = o3d.geometry.PointCloud()
            cond_pcd.points = o3d.utility.Vector3dVector(part_c[:,1:].cpu().numpy())
            cond_pcd.paint_uniform_color([0,1,0])
        
            print("node points", full_c.shape[0], "cond points", part_c.shape[0])
        
            # Visualization of random nodes and their connections
            self.vis_cache = [cond_pcd, nodes_pcd]
            
            if weights is not None:
                colors = np.array(cond_pcd.colors)
                rand_nodes = np.random.randint(0, full_c.shape[0], 10)
                
                # Color nodes and their matches
                for rand_node in rand_nodes:
                    colors[match_feats[rand_node].detach().cpu()] = weights[rand_node][:,None].detach().cpu() * np.array([1,0,1])
                
                cond_pcd.colors = o3d.utility.Vector3dVector(colors)
                nodes_color = np.array(nodes_pcd.colors)
                nodes_color[rand_nodes] = np.array([0,0,1])
                nodes_pcd.colors = o3d.utility.Vector3dVector(nodes_color)
            
            # Add coordinate frame
                mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
            self.vis_cache.append(mesh)
            
            # Add input point cloud if available
            if input_pcd is not None:
                input_points = x_node.slice(input_pcd).C.clone().float()
                input_points = (input_points + input_pcd.sparse().tensor_stride[0]/2) * self.node_voxel_size
                input_pcd_o3d = o3d.geometry.PointCloud()
                input_pcd_o3d.points = o3d.utility.Vector3dVector(input_points[:,1:].cpu().numpy())
                input_pcd_o3d.paint_uniform_color([0,1,1])
                self.vis_cache.insert(0, input_pcd_o3d)
            
                o3d.visualization.draw_geometries(self.vis_cache)

    def forward(self, x, part_feats, t):
        def process_stage(stage, prev_x, temp_emb, part_feats, x, is_up_stage=False):
            matches = self.match_part_to_full(prev_x, part_feats, stage=stage, input_pcd=x)
            p_feats = self.latents[stage](matches)
            t_feats = self.stage_temps[stage](temp_emb)
            batch_temp = torch.unique(prev_x.C[:, 0], return_counts=True)[1]
            t_feats = torch.repeat_interleave(t_feats, batch_temp, dim=0)
            w_feats = self.latemps[stage](torch.cat((p_feats, t_feats), -1))
            if is_up_stage:
                y = self.ups[stage - self.num_stages][0](prev_x * w_feats)
                y = ME.cat(y, x_feats[2 - (stage - self.num_stages)] if stage - self.num_stages < self.num_stages - 1 else x0)
                y = self.ups[stage - self.num_stages][1](y)
                return y
            else:
                return self.stages[stage](prev_x * w_feats)

        temp_emb = self.get_timestep_embedding(t)
        x0 = self.stem(x.sparse())
        prev_x = x0
        x_feats = [None] * self.num_stages

        for i in range(self.num_stages):
            if self.debug:
                print("stage", i)
            prev_x = process_stage(i, prev_x, temp_emb, part_feats, x)
            x_feats[i] = prev_x

        for i in range(self.num_stages):
            prev_x = process_stage(i + self.num_stages, prev_x, temp_emb, part_feats, x, is_up_stage=True)

        out = [self.last(prev_it) for prev_it in prev_x.slice(x).decomposed_features]
        return out
