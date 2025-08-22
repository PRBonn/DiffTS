import cugraph
import cupy
import numpy as np
import torch
from cudf import DataFrame


def cudf_edgelist_to_numpy(edge_list):
    return np.vstack((edge_list["src"].to_numpy(), edge_list["dst"].to_numpy())).T


def shortest_paths(root, edges, edge_weights, renumber=True):
    device = edges.device
    g = edge_graph(edges, edge_weights, renumber=renumber)
    r = cugraph.sssp(g, source=root)

    return (
        torch.as_tensor(r["vertex"], device=device).long(),
        torch.as_tensor(r["predecessor"], device=device).long(),
        torch.as_tensor(r["distance"], device=device),
    )


def graph_shortest_paths(root, graph, device):
    # device = edges.device

    r = cugraph.sssp(graph, source=root)
    return (
        torch.as_tensor(r["predecessor"], device=device).long(),
        torch.as_tensor(r["distance"], device=device),
    )


def pred_graph(preds, points):
    n = preds.shape[0]
    valid = preds >= 0

    dists = torch.norm(points - points[torch.clamp(preds, 0)], dim=1)
    dists[~valid] = 0.0

    edges = torch.stack([torch.arange(0, n, device=preds.device), preds], dim=1)
    edges[~valid, 1] = edges[~valid, 0]
    return edge_graph(edges, dists)


def pred_graph(verts, preds, points):
    n = preds.shape[0]
    valid = preds >= 0

    dists = torch.norm(points[verts] - points[torch.clamp(preds, 0)], dim=1)
    dists[~valid] = 0.0

    edges = torch.stack([torch.arange(0, n, device=preds.device), preds], dim=1)
    edges[~valid, 1] = edges[~valid, 0]
    return edge_graph(edges, dists)


def euclidean_distances(root, points, preds):
    g = pred_graph(preds, points)
    r = cugraph.sssp(g, source=root)
    return torch.as_tensor(r["distance"], device=points.device)


def edge_graph(edges, edge_weights, renumber=False):
    d = DataFrame()
    edges = cupy.asarray(edges)

    d["source"] = edges[:, 0]
    d["destination"] = edges[:, 1]
    d["weights"] = cupy.asarray(edge_weights)

    g = cugraph.Graph(directed=False)
    g.from_cudf_edgelist(d, edge_attr="weights", renumber=renumber)
    return g
