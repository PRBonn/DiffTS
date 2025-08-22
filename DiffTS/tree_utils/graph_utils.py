from __future__ import annotations

import cugraph
import frnn
import numpy as np
import torch
from tqdm import tqdm

from ..data_types.graph import Graph


def knn(src, dest, K=50, r=1.0, grid=None):
    src_lengths = src.new_tensor([src.shape[0]], dtype=torch.long)
    dest_lengths = src.new_tensor([dest.shape[0]], dtype=torch.long)
    dists, idxs, grid, _ = frnn.frnn_grid_points(
        src.unsqueeze(0),
        dest.unsqueeze(0),
        src_lengths,
        dest_lengths,
        K,
        r,
        return_nn=False,
        return_sorted=True,
    )

    return idxs.squeeze(0), dists.sqrt().squeeze(0), grid


def nn(src, dest, r=1.0, grid=None):
    idx, dist, grid = knn(src, dest, K=1, r=r, grid=grid)
    idx, dist = idx.squeeze(1), dist.squeeze(1)

    return idx, dist, grid


def nn_graph(points: torch.Tensor, radii, K=40):
    idxs, dists, _ = knn(points, points, K=K, r=radii.max().item())
    idxs[dists > radii.unsqueeze(1)] = -1
    edges, edge_weights = make_edges(dists, idxs)
    return Graph(points, edges, edge_weights)


def medial_nn_graph(points: torch.Tensor, radii, medial_dist, K=40):
    # edges weighted based on distance to medial axis
    idxs, dists, _ = knn(points, points, K=K, r=radii.max().item() / 4.0)
    dists_ = dists + medial_dist[idxs]  # Add medial distance to distance graph...
    idxs[dists > radii.unsqueeze(1)] = -1

    return make_edges(dists_, idxs)


def make_edges(dists, idxs):
    n = dists.shape[0]
    K = dists.shape[1]

    parent = torch.arange(n, device=dists.device).unsqueeze(1).expand(n, K)
    edges = torch.stack([parent, idxs], dim=2)

    valid = idxs.view(-1) > 0
    return edges.view(-1, 2)[valid], dists.view(-1)[valid]


def nn_flat(points, K=50, r=1.0, device=torch.device("cuda")):
    idxs, dists = nn(points, K, r, device)
    return idxs.reshape(-1), dists.reshape(-1)


def pcd_nn(points, radii, K=20):
    idxs, dists = nn_flat(np.asarray(points), K=K, r=float(radii.max() / 4.0))

    parent_vertex = np.repeat(np.arange(points.shape[0]), K)
    parent_vertex = parent_vertex.reshape(-1)

    edges = np.vstack((parent_vertex, idxs.reshape(-1))).T

    valid = idxs != -1
    return edges[valid], dists[valid]


def decompose_cuda_graph(cuda_graph, renumber_edges=False, device=torch.device("cuda")):
    pdf = cugraph.to_pandas_edgelist(cuda_graph)

    edges = torch.stack((torch.tensor(pdf["src"]), torch.tensor(pdf["dst"])), dim=1)
    edge_weights = torch.tensor(pdf["weights"])

    edges, edge_weights = edges.long().to(device), edge_weights.to(device)

    if renumber_edges:
        edges = remap_edges(edges)

    return edges, edge_weights


def remap_edges(edges):
    # Find unique node IDs and their corresponding indices
    unique_nodes, node_indices = torch.unique(edges, return_inverse=True)

    # Create a mapping from old node IDs to new node IDs
    mapping = torch.arange(unique_nodes.size(0), device=edges.device)

    # Map the old node IDs to new node IDs using the indices
    renumbered_edges = mapping[node_indices].reshape(edges.shape)

    return renumbered_edges


def connected_components(edges, edge_weights, minimum_vertices=0, max_components=10):
    g = cuda_graph(edges, edge_weights)

    connected_components = cugraph.connected_components(g)

    num_labels = connected_components["labels"].to_pandas().value_counts()
    valid_labels = num_labels[num_labels > minimum_vertices].index

    graphs = []

    for label in tqdm(
        valid_labels[:max_components], desc="Getting Connected Componenets", leave=False
    ):
        graphs.append(
            cugraph.subgraph(
                g, connected_components.query(f"labels == {label}")["vertex"]
            )
        )

    return graphs
