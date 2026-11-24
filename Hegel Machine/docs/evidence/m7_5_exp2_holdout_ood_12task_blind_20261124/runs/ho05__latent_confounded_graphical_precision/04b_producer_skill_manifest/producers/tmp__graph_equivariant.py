import numpy as np
from scipy.linalg import eigh
from graph_probe import generate
from deconfound_graph import _fit, _profile


def residual_scale(c):
    d, u = eigh(c, check_finite=False)
    w = u[:, :-2] ** 2
    return np.sqrt((w @ (1 / d[:-2])) / w.sum(axis=1))


def candidates(x):
    n, p = x.shape
    x = x - x.mean(axis=0)
    c = x.T @ x / n
    corr = c / np.sqrt(np.outer(c.diagonal(), c.diagonal()))
    eye = np.eye(p)
    penalty = .8775 / np.sqrt(n)
    ij = np.triu_indices(p, 1)
    def scores(s):
        return np.abs(s[ij]) / np.sqrt(s[ij[0], ij[0]] * s[ij[1], ij[1]]) * np.sqrt(n)
    def fit(c, scale, start=eye):
        return _fit(_profile(c * np.outer(scale, scale)), start, penalty=penalty)
    yield scores(fit(c, residual_scale(c)))
    d = residual_scale(corr)
    k = corr * np.outer(d, d)
    s = fit(corr, d)
    yield scores(s)
    r = np.sqrt(s.diagonal())
    yield scores(fit(k, r, s / np.outer(r, r)))
    for i in range(5):
        d = d * residual_scale(corr * np.outer(d, d)) ** .75
        if i in (1, 4):
            yield scores(fit(corr, d))
    yield scores(fit(corr, d, s * np.outer(residual_scale(corr) / d, residual_scale(corr) / d)))


def case(seed, mode):
    if mode not in ('hetero_coherent', 'factor2_weak'):
        x, a, _ = generate(seed, mode)
        return x, a
    _, a, s = generate(seed, 'hetero' if mode == 'hetero_coherent' else 'sign')
    rng = np.random.default_rng(seed + 800)
    b = rng.normal(size=(24, 2)) * .9
    if mode == 'hetero_coherent': b[:5] *= 2.5
    else: b[:, 1] *= .26
    return rng.normal(size=(900, 24)) @ np.linalg.cholesky(np.linalg.inv(s)).T + rng.normal(size=(900, 2)) @ b.T, a


def report(cases):
    ij = np.triu_indices(24, 1)
    results = []
    for x, a in cases:
        y = a[ij]
        q = np.array(list(candidates(x)))
        votes = q[:, None, :] >= np.array([2.3, 2.5, 2.7])[None, :, None]
        results.append(2 * (votes & y).sum(axis=2) / (y.sum() + votes.sum(axis=2)))
    v = np.array(results)
    return np.round(v.min(axis=0), 3).tolist(), np.round(v[:, :, 1].mean(axis=0), 3).tolist()


if __name__ == '__main__':
    z = np.load('/holdout_task/public_graph.npz')
    x, a = z['X'], z['gt_adj']
    rng = np.random.default_rng(42)
    print('methods: raw normalization, correlation, fit-rescale, spectral-rescale x2, spectral-rescale x5, spectral-rescale x5 warm; thresholds: 2.3, 2.5, 2.7')
    for mode, cases in [('public', [(x, a)]), ('public450', [(x[rng.permutation(len(x))[:450]], a) for _ in range(12)])]:
        print(mode, report(cases), flush=True)
    for mode in ('ring', 'sign', 'random', 'weak', 'hubs', 'coherent', 'hetero', 'weaker_factors', 'hetero_coherent', 'factor2_weak'):
        seeds = sorted(set(range(125, 149)) | ({37, 69, 96, 102, 106} if 'coherent' in mode else set()))
        print(mode, report(case(seed, mode) for seed in seeds), flush=True)
