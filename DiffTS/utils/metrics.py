import open3d as o3d
import numpy as np
import scipy

MESHTYPE = 6
TETRATYPE = 10
PCDTYPE = 1

class Metrics3D():
    def prediction_is_empty(self, geom):

        if isinstance(geom, o3d.geometry.Geometry):
            geom_type = geom.get_geometry_type().value
            if geom_type == MESHTYPE or geom_type == TETRATYPE:
                empty_v = self.is_empty(len(geom.vertices))
                empty_t = self.is_empty(len(geom.triangles))
                empty = empty_t or empty_v
            elif geom_type == PCDTYPE:
                empty = self.is_empty(len(geom.points))
            else:
                assert False, '{} geometry not supported'.format(geom.get_geometry_type())
        elif isinstance(geom, np.ndarray):
            empty = self.is_empty(len(geom[:, :3]))
        elif isinstance(geom, torch.Tensor):
            empty = self.is_empty(len(geom[:, :3]))
        else:
            assert False, '{} type not supported'.format(type(geom))

        return empty

    @staticmethod
    def convert_to_pcd(geom):

        if isinstance(geom, o3d.geometry.Geometry):
            geom_type = geom.get_geometry_type().value
            if geom_type == MESHTYPE or geom_type == TETRATYPE:
                geom_pcd = geom.sample_points_uniformly(1000000)
            elif geom_type == PCDTYPE:
                geom_pcd = geom
            else:
                assert False, '{} geometry not supported'.format(geom.get_geometry_type())
        elif isinstance(geom, np.ndarray):
            geom_pcd = o3d.geometry.PointCloud()
            geom_pcd.points = o3d.utility.Vector3dVector(geom[:, :3])
        elif isinstance(geom, torch.Tensor):
            geom = geom.detach().cpu().numpy()
            geom_pcd = o3d.geometry.PointCloud()
            geom_pcd.points = o3d.utility.Vector3dVector(geom[:, :3])
        else:
            assert False, '{} type not supported'.format(type(geom))

        return geom_pcd

    @staticmethod
    def is_empty(length):
        empty = True
        if length:
            empty = False
        return empty

        input()
class ChamferDistance():
    def __init__(self):
        self.pt_2_gt_dists = []
        self.gt_2_pt_dists = []
        self.combined_dists = []

        return

    def update(self, gt_pcd, pt_pcd):
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))
        dist_gt_2_pt = np.asarray(gt_pcd.compute_point_cloud_distance(pt_pcd))

        self.combined_dists.append((np.mean(dist_gt_2_pt) + np.mean(dist_pt_2_gt)) / 2)
        self.pt_2_gt_dists.append(np.mean(dist_pt_2_gt))
        self.gt_2_pt_dists.append(np.mean(dist_gt_2_pt))

    def reset(self):
        self.pt_2_gt_dists = []
        self.gt_2_pt_dists = []
        self.combined_dists = []

    def compute(self):
        combined_cdist = np.array(self.combined_dists).mean()
        pt_2_gt_cdist = np.array(self.pt_2_gt_dists).mean()
        gt_2_pt_cdist = np.array(self.gt_2_pt_dists).mean()
        return combined_cdist, pt_2_gt_cdist, gt_2_pt_cdist
class PrecisionRecall(Metrics3D):

    def __init__(self, min_t, max_t, num):
        self.thresholds = np.linspace(min_t, max_t, num)
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def update(self, gt_pcd, pt_pcd):
        # precision: predicted --> ground truth
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))

        # recall: ground truth --> predicted
        dist_gt_2_pt = np.asarray(gt_pcd.compute_point_cloud_distance(pt_pcd))

        for t in self.thresholds:
            p = np.where(dist_pt_2_gt < t)[0]
            p = 100 / len(dist_pt_2_gt) * len(p)
            self.pr_dict[t].append(p)

            r = np.where(dist_gt_2_pt < t)[0]
            r = 100 / len(dist_gt_2_pt) * len(r)
            self.re_dict[t].append(r)

            # fscore
            if p == 0 or r == 0:
                f = 0
            else:
                f = 2 * p * r / (p + r)
            self.f1_dict[t].append(f)

    def reset(self):
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def compute_at_threshold(self, threshold):
        t = self.find_nearest_threshold(threshold)
        pr = sum(self.pr_dict[t]) / len(self.pr_dict[t])
        re = sum(self.re_dict[t]) / len(self.re_dict[t])
        f1 = sum(self.f1_dict[t]) / len(self.f1_dict[t])
        return pr, re, f1, t

    def compute_auc(self):
        if len(self.pr_dict[self.thresholds[0]]) == 0:
            return 0, 0, 0
        dx = self.thresholds[1] - self.thresholds[0]
        perfect_predictor = scipy.integrate.simpson(np.ones_like(self.thresholds), dx=dx)

        pr, re, f1 = self.compute_at_all_thresholds()
        pr_area = scipy.integrate.simpson(pr, dx=dx)
        norm_pr_area = pr_area / perfect_predictor

        re_area = scipy.integrate.simpson(re, dx=dx)
        norm_re_area = re_area / perfect_predictor

        f1_area = scipy.integrate.simpson(f1, dx=dx)
        norm_f1_area = f1_area / perfect_predictor

        return norm_pr_area, norm_re_area, norm_f1_area

    def compute_at_all_thresholds(self):
        pr = [sum(self.pr_dict[t]) / len(self.pr_dict[t]) for t in self.thresholds]
        re = [sum(self.re_dict[t]) / len(self.re_dict[t]) for t in self.thresholds]
        f1 = [sum(self.f1_dict[t]) / len(self.f1_dict[t]) for t in self.thresholds]
        return pr, re, f1

    def find_nearest_threshold(self, value):
        idx = (np.abs(self.thresholds - value)).argmin()
        return self.thresholds[idx]

