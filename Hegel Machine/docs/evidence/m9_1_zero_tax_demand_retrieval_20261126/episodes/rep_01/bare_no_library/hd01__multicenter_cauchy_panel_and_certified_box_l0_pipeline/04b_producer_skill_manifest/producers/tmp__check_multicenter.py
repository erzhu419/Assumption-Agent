import importlib.util
import pickle
import time
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

base = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('pipeline', base / 'multicenter_certified_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def check_cert(data, result):
    x, y, r, a, m = (data[key] for key in ('X', 'y', 'lam2', 'lam0', 'M'))
    w = result['weights']
    p = x.shape[1]
    g = x.T @ x
    b = x.T @ y
    eps = min(1e-8, m)
    val = .5 * (np.sum((x @ w - y)**2) + r * np.sum(w*w)) + a * np.sum(abs(w) > 1e-8)
    assert w.shape == (p,) and np.max(abs(w)) <= m
    assert abs(val - result['objective']) < 1e-10
    leaves = result['subcube_leaves']
    assert 1 <= len(leaves) <= 2048, len(leaves)
    seen = []
    bounds = []
    volume = 0
    for leaf in leaves:
        z, s = map(set, (leaf['fixed_zero'], leaf['fixed_active']))
        assert not z & s and z | s <= set(range(p))
        assert all(z & si or s & zi for zi, si in seen), 'overlapping cubes'
        seen.append((z, s))
        volume += 1 << (p-len(z)-len(s))
        d, v = leaf['minorant_diagonal'], leaf['minorant_center']
        assert np.all(d >= 0)
        rem = g - np.diag(d)
        assert np.linalg.eigvalsh(rem)[0] >= -1e-9
        ell = rem @ v - b
        q = d + r
        def quadratic_at(t):
            return .5 * q * t*t + ell*t
        off = quadratic_at(np.clip(-ell/q, -eps, eps))
        on = a + quadratic_at(-np.where(ell >= 0, 1, -1) * np.clip(abs(ell)/q, eps, m))
        sub = sum(off[j] if j in z else on[j] if j in s else min(off[j], on[j]) for j in range(p))
        expected = .5 * (y @ y - v @ rem @ v) + sub
        assert leaf['lb'] <= expected + 1e-11
        bounds.append(leaf['lb'])
    assert volume == 1 << p
    lb = result['certified_lower_bound']
    assert abs(lb - min(bounds)) <= 1e-10
    return val - lb, len(leaves)

for i, row in enumerate(pickle.load(open(base / 'public_multicenter_l0_studies.pkl', 'rb'))):
    study, gt = row['study'], row['gt']
    c = study['cohort']; centers = np.unique(c['center_id']); z = np.empty_like(c['X_raw']); weights = np.zeros(len(z))
    for center in centers:
        rows = c['center_id'] == center
        col = [v[np.isfinite(v)] for v in c['controls'][rows].T]
        valid = [v for v in col if len(v) >= 2]
        z[rows] = (c['X_raw'][rows] - np.median([v.mean() for v in valid])) / np.median([v.std() for v in valid])
        for stratum in range(3):
            cell = rows & (c['stratum'] == stratum)
            weights[cell] = 1 / cell.sum()
    for j in range(z.shape[1]):
        mask = np.isfinite(z[:, j]); z[~mask, j] = np.average(z[mask, j], weights=weights[mask])
    res = mod.solve_multicenter_certified_l0(study)
    assert np.allclose(res['aligned_biomarkers'], z, rtol=1e-13, atol=1e-13)
    mae = np.mean(abs(z - gt['Z_true']))
    slope = np.linalg.norm(res['robust_potency_weights'] - gt['beta_potency']) / np.linalg.norm(gt['beta_potency'])
    cert = res['certified_selection']
    gap, k = check_cert(study['sparse_selection'], cert)
    assert abs(cert['objective'] - gt['exact_l0_objective']) < 1e-8 and 0 <= gap <= 1e-5 and slope <= .08
    print('public', i, 'MAE', mae, 'slope', slope, 'gap', gap, 'leaves', k, flush=True)

rng = np.random.default_rng(211)
for case in ('orthogonal-near-knee','orthogonal-binding-box','embedded-correlated-box','dense-public-condition','dense-collinear'):
    p = 22; r = .14; a = .22; m = .35 if 'box' in case else 1.1
    x, _ = np.linalg.qr(rng.normal(size=(44,p)))
    b = rng.uniform(-1, 1, size=p)
    if case == 'orthogonal-near-knee':
        b = np.sqrt(2*a*(1+r)) * rng.choice([-1,1], p) * (1 + rng.uniform(-2e-5, 2e-5, p))
    if case == 'embedded-correlated-box':
        x[:, :8] = x[:, :8] @ np.linalg.cholesky(np.eye(8)*.65 + .35)
        b[8:] *= .1
    if case.startswith('dense'):
        u, _ = np.linalg.qr(rng.normal(size=(p,p)))
        lo = .5 if case == 'dense-public-condition' else .02
        x = x @ ((u * np.sqrt(np.linspace(lo, 1.5, p))) @ u.T)
        b *= .9
    y = np.linalg.lstsq(x.T, b, rcond=None)[0]
    data = dict(X=x, y=y, lam2=r, lam0=a, M=m)
    t = time.monotonic(); out = mod._sparse(data); dt = time.monotonic()-t
    gap,k = check_cert(data,out)
    print(case, 'gap', gap, 'leaves', k, 'bound hits', np.sum(abs(out['weights']) == m), 'seconds', dt, flush=True)
    if case.startswith('orthogonal'):
        c = np.clip(b/(1+r), -m,m)
        expected = y@y/2 + np.minimum(0, (1+r)*c*c/2-b*c+a).sum()
        assert abs(out['objective'] - expected) < 1e-6
    if case == 'embedded-correlated-box':
        expected = y@y/2
        for mask in range(1,256):
            ind = [j for j in range(8) if mask >> j & 1]
            solver = lsq_linear(np.vstack([x[:,ind],np.sqrt(r)*np.eye(len(ind))]), np.r_[y,np.zeros(len(ind))], bounds=(-m,m), tol=1e-12)
            expected = min(expected, solver.cost + a*len(ind))
        assert abs(out['objective'] - expected) < 1e-6
    assert 0 <= gap <= 1e-5
