import numpy as np
from scipy.linalg import eigh
from graph_probe import generate
from deconfound_graph import _fit, _profile

modes = ('ring', 'sign', 'random', 'weak', 'hubs', 'coherent', 'hetero', 'weaker_factors', 'hetero_coherent')

def estimate(x, method):
    n, p = x.shape
    z = x - x.mean(0)
    c = z.T @ z / n
    if method != 'original':
        sd = np.sqrt(c.diagonal())
        c = c / np.outer(sd, sd)
    d, u = eigh(c, check_finite=False)
    w = u[:, :-2] ** 2
    scale = np.sqrt((w @ (1 / d[:-2])) / w.sum(1))
    if method == 'corr_inverse_tail':
        scale = np.sqrt(w @ (1 / d[:-2]))
    elif method == 'corr_residual_var':
        scale = np.sqrt(w.sum(1) / (w @ d[:-2]))
    k = c * np.outer(scale, scale)
    s = _fit(_profile(k), np.eye(p), penalty=.8775 / np.sqrt(n))
    if method == 'corr_iterated':
        r = np.sqrt(s.diagonal())
        s = _fit(_profile(k * np.outer(r, r)), s / np.outer(r, r), penalty=.8775 / np.sqrt(n))
    q = np.abs(s) / np.sqrt(np.outer(s.diagonal(), s.diagonal()))
    b = q >= 2.5 / np.sqrt(n)
    np.fill_diagonal(b, False)
    return b

def case(seed, mode):
    if mode != 'hetero_coherent':
        return generate(seed, mode)[:2]
    _, y, t = generate(seed, 'hetero')
    rng = np.random.default_rng(seed + 800)
    b = rng.normal(size=(24, 2)) * .9
    b[:5] *= 2.5
    x = rng.normal(size=(900,24)) @ np.linalg.cholesky(np.linalg.inv(t)).T + rng.normal(size=(900,2)) @ b.T
    return x, y

def stress(x, y):
    row = []
    for name in methods:
        b = estimate(x, name)
        row.append(2 * np.count_nonzero(b & y) / (b.sum() + y.sum()))
    return row

methods = ('original', 'corr', 'corr_inverse_tail', 'corr_residual_var', 'corr_iterated')
z = np.load('/holdout_task/public_graph.npz'); x, y = z['X'], z['gt_adj']
rng = np.random.default_rng(42); perm = rng.permutation(24); factors = np.exp(rng.uniform(-.75,.75,24))
print('methods', methods)
print('public F1', np.round(stress(x, y), 4))
print('public transformed adjacency differences', [np.count_nonzero(estimate(x[:,perm]*factors+3, name) != estimate(x, name)[np.ix_(perm,perm)]) for name in methods])
for mode in modes:
    seeds = sorted(set(range(90,114)) | ({50, 69, 83, 63, 33, 37} if 'coherent' in mode else set()))
    vals = np.array([stress(*case(seed, mode)) for seed in seeds]); print(mode, 'min', np.round(vals.min(0), 4), 'mean', np.round(vals.mean(0), 4), 'worst seeds', np.array(seeds)[vals.argmin(0)], flush=True)
