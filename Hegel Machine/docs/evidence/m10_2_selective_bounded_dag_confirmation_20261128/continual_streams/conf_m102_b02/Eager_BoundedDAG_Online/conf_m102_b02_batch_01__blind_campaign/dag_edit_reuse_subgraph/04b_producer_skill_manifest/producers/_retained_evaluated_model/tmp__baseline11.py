import numpy as np


def _latent(study):
    a = study['panel_a']
    u = np.asarray(a['u'], dtype=float)
    v = np.asarray(a['v'], dtype=float)
    design = np.c_[np.ones(len(u)), u]
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        coef = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
        residual = np.linalg.norm(design @ coef - v, axis=1)
        proposed = residual < max(0.001, 3 * np.median(residual))
        if proposed.sum() < design.shape[1] or np.array_equal(proposed, keep):
            break
        keep = proposed
    x = np.vstack((study['train']['x'], study['test']['x']))
    return np.linalg.solve(coef[1:].T, (x - coef[0]).T).T


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    z = _latent(study) * 0.5
    n = len(study['train']['x'])
    return {'z': z[:n].copy(), 'zt': z[n:].copy()}
