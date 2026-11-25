import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score

spec = importlib.util.spec_from_file_location('assay', '/holdout_task/multicenter_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
nodes, weights = hermgauss(20)
nodes *= np.sqrt(2.)
weights /= np.sqrt(np.pi)

def marginal(eta, missing, beta):
    sd = np.sqrt(missing @ (beta * beta))
    pp = expit(eta[:, None] + sd[:, None] * nodes)
    return pp @ weights, pp, sd

def fit(z, s, c, y, missing):
    base = m._design(z, s)
    effects = c[:, None] == np.unique(c)
    X = np.column_stack((base, effects))
    nb = base.shape[1]
    d = z.shape[1]
    ridge = np.r_[0., np.full(d, 3.), .25, .25, np.full(effects.shape[1], 25.)]
    init = np.r_[m._fit(z, s, c, y), np.zeros(effects.shape[1])]
    def fg(b):
        beta = b[1:d+1]
        p, pp, sd = marginal(X @ b, missing, beta)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        dp = pp * (1. - pp)
        a = (p - y) / (p * (1. - p))
        g = X.T @ (a * (dp @ weights)) + ridge * b
        k = np.divide(a * (dp @ (weights * nodes)), sd, out=np.zeros_like(sd), where=sd > 0.)
        g[1:d+1] += beta * (missing.T @ k)
        return -np.sum(np.where(y == 1, np.log(p), np.log1p(-p))) + .5 * ((ridge * b) @ b), g
    result = minimize(fg, init, jac=True, method='L-BFGS-B', options={'ftol': 1e-12})
    assert result.success, result.message
    return result.x[:nb]

def score(y, p):
    return np.array([roc_auc_score(y, p), brier_score_loss(y, p), balanced_accuracy_score(y, p >= .5)])

with open('/holdout_task/public_multicenter_assay.pkl', 'rb') as f:
    entries = pickle.load(f)['cohorts']
for j, entry in enumerate(entries):
    d, g = entry['dataset'], entry['gt']
    t, q, r = (d[k] for k in ('train', 'query', 'reference_controls'))
    panel, pooled = m._reference_transforms(r)
    z, zq = (m._align(a, panel, pooled) for a in (t, q))
    c, s, y = (t[k] for k in ('center_id', 'stratum', 'y'))
    miss, missq = np.isnan(t['X']), np.isnan(q['X'])
    results = []
    for center in np.unique(c):
        h = c == center
        b = fit(z[~h], s[~h], c[~h], y[~h], miss[~h])
        b0 = m._fit(z[~h], s[~h], c[~h], y[~h])
        Xh = m._design(z[h], s[h])
        results.append([score(y[h], expit(Xh @ b0)), score(y[h], marginal(Xh @ b, miss[h], b[1:1+z.shape[1]])[0])])
    a = np.asarray(results)
    print('cohort', j, 'LOCO baseline/marginal, mean', a.mean(0).round(4), 'worst', np.stack((a[:,:,0].min(0), a[:,:,1].max(0), a[:,:,2].min(0)), axis=1).round(4))
    b = fit(z, s, c, y, miss)
    p = marginal(m._design(zq, q['stratum']) @ b, missq, b[1:1+z.shape[1]])[0]
    p0 = m.calibrate_multicenter_assay(d)['query_probabilities']
    for center in np.unique(q['center_id']):
        h = q['center_id'] == center
        yq = g['y_query'][h]
        print(' query', center, 'baseline/marginal/oracle', np.stack([score(yq, v[h]) for v in (p0, p, g['prob_query'])]).round(4))
