from typing import List

import cugraph
import cupy
import torch
from cugraph import sssp
from tqdm import tqdm

from ..data_types.cloud import Cloud
from ..data_types.graph import Graph
from ..data_types.tree import DisjointTreeSkeleton, TreeSkeleton
from .filter import outlier_removal
from .graph import decompose_cuda_graph, nn_graph
from .path import sample_tree
from .shortest_path import pred_graph, shortest_paths


class Skeletonizer:
    def __init__(
        self,
        K: int,
        min_connection_length: float,
        minimum_graph_vertices: int,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.K = K
        self.min_connection_length = min_connection_length
        self.minimum_graph_vertices = minimum_graph_vertices
        self.device = device

    def forward(self, cloud: Cloud) -> DisjointTreeSkeleton:
        cloud = cloud.to_device(self.device)
        # mask = outlier_removal(cloud.medial_pts, cloud.radius.unsqueeze(1), nb_points=8)
        # cloud = cloud.filter(mask)

        # graph: Graph = nn_graph(
        #     cloud.medial_pts,
        #     cloud.radius.clamp(min=self.min_connection_length),
        #     K=self.K,
        # )
        graph: Graph = nn_graph(
            cloud.xyz,
            torch.full((cloud.xyz.shape[0],), 0.1, device=self.device),
            K=self.K,
        )

        subgraphs: List[cugraph.Graph] = graph.connected_cugraph_components(
            minimum_vertices=self.minimum_graph_vertices
        )

        skeletons = []
        for subgraph_id, subgraph in enumerate(
            tqdm(subgraphs, desc="Processing Connected Components", leave=False)
        ):
            skeletons.append(
                self.process_subgraph(cloud, subgraph, skeleton_id=subgraph_id)
            )

        return DisjointTreeSkeleton(skeletons)

    def process_subgraph(self, cloud, subgraph, skeleton_id=0) -> TreeSkeleton:
        """Extract skeleton for connected component"""

        subgraph_vertice_idx = torch.tensor(
            cupy.unique(subgraph.edges().values),
            device=self.device,
        )

        subgraph_cloud = cloud.filter(subgraph_vertice_idx)

        edges, edge_weights = decompose_cuda_graph(
            subgraph,
            renumber_edges=True,
            device=self.device,
        )

        verts, preds, distance = shortest_paths(
            subgraph_cloud.root_idx,
            edges,
            edge_weights,
            renumber=False,
        )

        predecessor_graph = pred_graph(verts, preds, subgraph_cloud.medial_pts)

        distances = torch.as_tensor(
            sssp(predecessor_graph, source=subgraph_cloud.root_idx)["distance"],
            device=self.device,
        )

        branches = sample_tree(
            subgraph_cloud.medial_pts,
            subgraph_cloud.radius.unsqueeze(1),
            preds,
            distances,
            subgraph_cloud.xyz,
        )

        return TreeSkeleton(skeleton_id, branches)

    @staticmethod
    def from_cfg(cfg):
        return Skeletonizer(
            K=cfg.K,
            min_connection_length=cfg.min_connection_length,
            minimum_graph_vertices=cfg.minimum_graph_vertices,
            edge_non_linear=cfg.edge_non_linear,
        )


if __name__ == "__main__":
    main()
