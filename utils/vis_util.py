import numpy as np


def make_grid_visualization(vis, grid=8, max_bz=8):
    """
    vis: (N, H, W, C), uint8 or float
    returns: (B, H_grid, W_grid, C)
    """
    assert vis.ndim == 4
    n, h, w, c = vis.shape
    col = grid
    row = min(grid, max(1, n // col))
    if n % (col * row) != 0:
        n = min(n, col * row * max_bz)
        vis = vis[:n]
        n, h, w, c = vis.shape
    if n % (col * row) != 0:
        pad = col * row - (n % (col * row))
        vis = np.concatenate([vis, np.repeat(vis[-1:], pad, axis=0)], axis=0)
        n = vis.shape[0]

    vis = vis.reshape((-1, col, row * h, w, c))
    vis = np.einsum("mlhwc->mhlwc", vis)
    vis = vis.reshape((-1, row * h, col * w, c))
    bz = min(vis.shape[0], max_bz)
    return vis[:bz]
