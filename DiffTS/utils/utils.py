import torch
import numpy as np
from DiffTS.utils.scheduling import beta_func
import open3d as o3d

def compute_diffusion_params(params):
    # alphas and betas
    if params['diff']['beta_func'] == 'cosine':
        betas = beta_func[params['diff']['beta_func']](params['diff']['t_steps'])
    else:
        betas = beta_func[params['diff']['beta_func']](
                params['diff']['t_steps'],
                params['diff']['beta_start'],
                params['diff']['beta_end'],
        )

    t_steps = params['diff']['t_steps']
    s_steps = params['diff']['s_steps']
    alphas = 1. - betas
    alphas_cumprod = torch.tensor(
        np.cumprod(alphas, axis=0), dtype=torch.float32, device=torch.device('cuda')
    )

    alphas_cumprod_prev = torch.tensor(
        np.append(1., alphas_cumprod[:-1].cpu().numpy()), dtype=torch.float32, device=torch.device('cuda')
    )

    betas = torch.tensor(betas, device=torch.device('cuda'))
    alphas = torch.tensor(alphas, device=torch.device('cuda'))

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod) 
    sqrt_recip_alphas = torch.sqrt(1. / alphas)
    sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1.)

    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    sqrt_posterior_variance = torch.sqrt(posterior_variance)
    posterior_log_var = torch.log(
        torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance))
    )

    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
    
    return {
        't_steps': t_steps,
        's_steps': s_steps,
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'log_one_minus_alphas_cumprod': log_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_recip_alphas_cumprod': sqrt_recip_alphas_cumprod,
        'sqrt_recipm1_alphas_cumprod': sqrt_recipm1_alphas_cumprod,
        'posterior_variance': posterior_variance,
        'sqrt_posterior_variance': sqrt_posterior_variance,
        'posterior_log_var': posterior_log_var,
        'posterior_mean_coef1': posterior_mean_coef1,
        'posterior_mean_coef2': posterior_mean_coef2,
    }
    
def normalize_vecs(vecs):
    vec_norms = vecs.norm(dim=-1, keepdim=True)
    vec_norms[vec_norms == 0] = 1
    vecs = vecs / vec_norms
    return vecs

def o3d_fps_sampling(points, num_samples):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    o3d_pcd = o3d_pcd.farthest_point_down_sample(num_samples)
    sampled_points = torch.tensor(np.array(o3d_pcd.points), device=points.device)
    return sampled_points
    
def visualize_step_t(x_t, gt_pts, pcd, max_range, pidx=0):
    points = x_t.F.detach().cpu().numpy()
    points = points.reshape(gt_pts.shape[0],-1,3)
    obj_mean = gt_pts.mean(axis=-2)
    points = np.concatenate((points[pidx], gt_pts[pidx]), axis=0)

    dist_pts = np.sqrt(np.sum((points - obj_mean[pidx])**2, axis=-1))
    dist_idx = dist_pts < max_range

    full_pcd = len(points) - len(gt_pts[pidx])
    print(f'\n[{dist_idx.sum() - full_pcd}|{dist_idx.shape[0] - full_pcd }] points inside margin...')

    pcd.points = o3d.utility.Vector3dVector(points[dist_idx])
    
    colors = np.ones((len(points), 3)) * .5
    colors[:len(gt_pts[0])] = [1.,.3,.3]
    colors[-len(gt_pts[0]):] = [.3,1.,.3]
    pcd.colors = o3d.utility.Vector3dVector(colors[dist_idx])
    
def vis_pts_tensor(points_list):
    pcds = []
    for i, points in enumerate(points_list):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        # random color
        pcd.paint_uniform_color(np.random.rand(3))
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)
    
def vis_flow_vecs(nodes, parents, len=0.01):
    directions = parents - nodes
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    directions *= len  # scale for visualization
    tips = nodes + directions
    flow_vecs = o3d.geometry.LineSet()
    flow_vecs.points = o3d.utility.Vector3dVector(np.concatenate((nodes, tips), axis=0))
    flow_vecs.lines = o3d.utility.Vector2iVector(np.stack((np.arange(nodes.shape[0]), np.arange(nodes.shape[0], nodes.shape[0] + tips.shape[0])), axis=1))
    return flow_vecs