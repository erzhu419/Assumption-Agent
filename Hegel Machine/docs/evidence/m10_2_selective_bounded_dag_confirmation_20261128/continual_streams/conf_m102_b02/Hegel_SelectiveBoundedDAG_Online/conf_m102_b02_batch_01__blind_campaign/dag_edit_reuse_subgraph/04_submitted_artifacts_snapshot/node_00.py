"""Robustly recover latent input coordinates from panel A."""
import numpy as np


def _latent(study):
    panel = study['panel_a']
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    d = np.column_stack((np.ones(len(u)), u))
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        coef = np.linalg.lstsq(d[keep], v[keep], rcond=None)[0]
        e = np.linalg.norm(d @ coef - v, axis=1)
        nxt = e <= max(0.001, 3 * np.median(e))
        if np.array_equal(keep, nxt) or nxt.sum() < d.shape[1]:
            break
        keep = nxt
    coef = np.linalg.lstsq(d[keep], v[keep], rcond=None)[0]
    x, xt = (np.asarray(study[k]['x'], dtype=float) for k in ('train', 'test'))
    z = np.linalg.solve(coef[1:].T, (np.vstack((x, xt)) - coef[0]).T).T
    return z, len(x)


def run_node(inputs: dict) -> dict:
    z, n = _latent(inputs['__input__'])
    z *= 0.2
    return {'z': z[:n].copy(), 'zt': z[n:].copy()}
