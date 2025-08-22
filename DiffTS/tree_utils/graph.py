from dataclasses import dataclass
from typing import List

# import cugraph
# import cupy
import open3d as o3d
import torch
from cudf import DataFrame
from torchtyping import TensorType
from tqdm import tqdm

from ..o3d_abstractions.geometries import o3d_line_set


@dataclass
class Graph:
    vertices: TensorType["N", 3]
    edges: TensorType["N", 2]
    edge_weights: TensorType["N", 1]

    def to_o3d_lineset(self, colour=(1, 0, 0)) -> o3d.geometry.LineSet:
        graph_cpu = self.to_device(torch.device("cpu"))
        return o3d_line_set(graph_cpu.vertices, graph_cpu.edges, colour=colour)

    def to_device(self, device: torch.device):
        return Graph(
            self.vertices.to(device),
            self.edges.to(device),
            self.edge_weights.to(device),
        )

    def connected_cugraph_components(self, minimum_vertices=10) -> List[cugraph.Graph]:

        g = cuda_graph(self.edges, self.edge_weights)

        df = cugraph.connected_components(g)

        components = []
        for label in tqdm(
            df["labels"].unique().to_pandas(),
            desc="Finding Connected Components",
            leave=False,
        ):
            subgraph_vertices = df[df["labels"] == label]["vertex"]

            if subgraph_vertices.count() < minimum_vertices:
                continue

            components.append(cugraph.subgraph(g, subgraph_vertices))

        return sorted(components, key=lambda graph: len(graph.nodes()), reverse=True)


def cuda_graph(edges, edge_weights, renumber=False):

    edges = cupy.asarray(edges)
    edge_weights = cupy.asarray(edge_weights)

    d = DataFrame()
    d["source"] = edges[:, 0]
    d["destination"] = edges[:, 1]
    d["weights"] = edge_weights
    g = cugraph.Graph(directed=False)
    g.from_cudf_edgelist(d, edge_attr="weights", renumber=renumber)

    return g
