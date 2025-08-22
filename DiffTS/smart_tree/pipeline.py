from pathlib import Path

import numpy as np
import torch

from .data_types.cloud import Cloud
from .data_types.tree import DisjointTreeSkeleton
from .o3d_abstractions.visualizer import o3d_viewer
from .util.file import (load_cloud, save_o3d_cloud,
                        save_o3d_lineset, save_o3d_mesh)


class Pipeline:
    def __init__(
        self,
        preprocessing,
        model_inference,
        skeletonizer,
        repair_skeletons=False,
        smooth_skeletons=False,
        smooth_kernel_size=0,
        prune_skeletons=False,
        min_skeleton_radius=0.0,
        min_skeleton_length=1000,
        view_model_output=False,
        view_skeletons=False,
        save_outputs=False,
        save_path="/",
        branch_classes=[0],
        cmap=[[1, 0, 0], [0, 1, 0]],
        device=torch.device("cuda:0"),
    ):
        self.preprocessing = preprocessing
        self.model_inference = model_inference
        self.skeletonizer = skeletonizer

        self.repair_skeletons = repair_skeletons
        self.smooth_skeletons = smooth_skeletons
        self.smooth_kernel_size = smooth_kernel_size
        self.prune_skeletons = prune_skeletons

        self.min_skeleton_radius = min_skeleton_radius
        self.min_skeleton_length = min_skeleton_length

        self.view_model_output = view_model_output
        self.view_skeletons = view_skeletons

        self.save_outputs = save_outputs
        self.save_path = save_path

        self.branch_classes = branch_classes
        self.cmap = np.asarray(cmap)
        self.device = device

    def process_cloud(self, path: Path =None, cloud: Cloud=None):
        # Load point cloud
        cloud: Cloud = load_cloud(path) if path != None else cloud
        
        cloud = cloud.to_device(self.device)
        cloud = self.preprocessing(cloud)

        # Run point cloud through model to predict class, radius, direction
        lc: Cloud = self.model_inference.forward(cloud).to_device(self.device)
        if self.view_model_output:
            lc.view(self.cmap)

        # Filter only the branch points for skeletonizaiton
        branch_cloud: Cloud = lc.filter_by_class(self.branch_classes)

        # Run the branch cloud through skeletonization algorithm, then post process
        skeleton: DisjointTreeSkeleton = self.skeletonizer.forward(branch_cloud)

        self.post_process(skeleton)

        # View skeletonization results
        if self.view_skeletons:
            o3d_viewer(
                [
                    skeleton.to_o3d_tube(),
                    skeleton.to_o3d_lineset(),
                    skeleton.to_o3d_tube(colour=False),
                    cloud.to_o3d_cld(),
                ],
                line_width=5,
            )

        if self.save_outputs:
            print("Saving Outputs")
            sp = self.save_path
            save_o3d_lineset(f"{sp}/skeleton.ply", skeleton.to_o3d_lineset())
            save_o3d_mesh(f"{sp}/mesh.ply", skeleton.to_o3d_tube())
            save_o3d_cloud(f"{sp}/cloud.ply", lc.to_o3d_cld())
            save_o3d_cloud(f"{sp}/seg_cld.ply", lc.to_o3d_seg_cld(self.cmap))

    def post_process(self, skeleton: DisjointTreeSkeleton):
        if self.prune_skeletons:
            skeleton.prune(
                min_length=self.min_skeleton_length,
                min_radius=self.min_skeleton_radius,
            )

        if self.repair_skeletons:
            skeleton.repair()

        if self.smooth_skeletons:
            skeleton.smooth(self.smooth_kernel_size)

    @staticmethod
    def from_cfg(inferer, skeletonizer, cfg):
        return Pipeline(
            inferer,
            skeletonizer,
            preprocessing_cfg=cfg.preprocessing,
            repair_skeletons=cfg.repair_skeletons,
            smooth_skeletons=cfg.smooth_skeletons,
            smooth_kernel_size=cfg.smooth_kernel_size,
            prune_skeletons=cfg.prune_skeletons,
            min_skeleton_radius=cfg.min_skeleton_radius,
            min_skeleton_length=cfg.min_skeleton_length,
            view_model_output=cfg.view_model_output,
            view_skeletons=cfg.view_skeletons,
            save_outputs=cfg.save_outputs,
            branch_classes=cfg.branch_classes,
            cmap=cfg.cmap,
        )
