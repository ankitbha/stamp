import numpy as np

# =============================================================================
# Graph construction
# =============================================================================

def build_graph_from_xy(sensors_xy: np.ndarray, mode: str, knn_k: int) -> np.ndarray:
    """
    Build directed edge_index [2, E] with edges j->i (source->target).
    """
    xy = sensors_xy.astype(np.float64)
    S = xy.shape[0]

    # Pairwise distances
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)

    edges_src = []
    edges_dst = []

    if mode == "fully_connected":
        for i in range(S):
            for j in range(S):
                if i == j:
                    continue
                edges_src.append(j)
                edges_dst.append(i)

    elif mode == "knn":
        k = int(knn_k)
        for i in range(S):
            nn = np.argsort(d2[i])[:k]
            for j in nn:
                edges_src.append(int(j))
                edges_dst.append(int(i))
    else:
        raise ValueError(f"Unknown graph mode: {mode}")

    edge_index = np.stack([np.array(edges_src, dtype=np.int64), np.array(edges_dst, dtype=np.int64)], axis=0)
    return edge_index


def rbf_encode(r: np.ndarray, num: int, rmax: float) -> np.ndarray:
    # r: [E]
    # centers on [0, rmax]
    centers = np.linspace(0.0, float(rmax), int(num), dtype=np.float64)
    # bandwidth: spacing
    if len(centers) > 1:
        bw = (centers[1] - centers[0]) + 1e-12
    else:
        bw = float(rmax) + 1e-12
    # Gaussian RBF
    return np.exp(-0.5 * ((r[:, None] - centers[None, :]) / bw) ** 2).astype(np.float32)


def build_edge_features(
    sensors_xy: np.ndarray,
    edge_index: np.ndarray,
    edge_eps: float,
    use_rbf: bool,
    rbf_num: int,
    rbf_rmax: float,
) -> np.ndarray:
    """
    Edge features (data-only geometry): [dx, dy, r, 1/r, (optional RBF(r))]
    where dx,dy are (x_src - x_dst), (y_src - y_dst) for edge src->dst.
    """
    xy = sensors_xy.astype(np.float64)
    src = edge_index[0]
    dst = edge_index[1]

    dx = xy[src, 0] - xy[dst, 0]
    dy = xy[src, 1] - xy[dst, 1]
    r = np.sqrt(dx * dx + dy * dy) + 1e-12
    invr = 1.0 / (r + float(edge_eps))
    invr = np.clip(invr, 0.0, 20.0)

    feats = [dx.astype(np.float32), dy.astype(np.float32), r.astype(np.float32), invr.astype(np.float32)]
    edge_attr = np.stack(feats, axis=1)

    if use_rbf:
        edge_attr = np.concatenate([edge_attr, rbf_encode(r, num=rbf_num, rmax=rbf_rmax)], axis=1)

    edge_attr = (edge_attr - edge_attr.mean(0)) / (edge_attr.std(0) + 1e-6)


    return edge_attr.astype(np.float32)
