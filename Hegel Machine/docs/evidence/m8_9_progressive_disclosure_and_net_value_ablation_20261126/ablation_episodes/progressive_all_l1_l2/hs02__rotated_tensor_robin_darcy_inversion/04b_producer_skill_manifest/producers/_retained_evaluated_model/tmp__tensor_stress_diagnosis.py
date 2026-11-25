import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from tensor_darcy_robin import _responses, _robust_sources

with open('/holdout_task/public_tensor_cases.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']

for case_id, case in enumerate(cases):
    p, g = case['problem'], case['gt']
    ij = p['sensor_ij']
    Z = _responses(p)
    X = Z[tuple(ij.T)][:, 1:]
    n, m = X.shape
    a = p['tikhonov_alpha']
    w0, o0 = g['source_weights'], g['sensor_offset']
    mu = X @ w0 + o0
    D = np.column_stack((X, np.ones(n)))
    penalty = np.column_stack((np.sqrt(a) * np.eye(m), np.zeros(m)))
    U = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)[0]
    leverage = np.sum(U**2, axis=1)
    rng = np.random.default_rng(269 + case_id)
    errors, oracle_errors, gaps = [], [], []
    for trial in range(240):
        k = (5, 6, 8, 10)[trial % 4]
        mode = trial // 4 % 5
        if mode == 0:
            bad = rng.choice(n, k, replace=False)
        elif mode == 1:
            bad = np.argsort(leverage)[-k:]
        elif mode == 2:
            bad = np.argsort(np.sum((ij - ij[rng.integers(n)])**2, axis=1))[:k]
        else:
            direction = U @ rng.normal(size=m)
            bad = np.argsort(direction if mode == 3 else abs(direction))[-k:]
        keep = np.ones(n, dtype=bool)
        keep[bad] = False
        clean = mu + rng.normal(0, .0015, n)
        y = clean.copy()
        signs = rng.choice([-1, 1], k) if mode < 3 else np.sign(direction[bad])
        y[bad] += signs * rng.uniform(1.5, 4, k)
        w, o = _robust_sources(X, y, a)
        oracle = np.linalg.lstsq(np.vstack((D[keep], penalty)), np.r_[y[keep], np.zeros(m)], rcond=None)[0]
        def metrics(w, o):
            dw = w - w0
            return np.array([np.linalg.norm(dw) / np.linalg.norm(w0), abs(o - o0), np.max(abs(Z[..., 1:] @ dw))])
        errors.append(metrics(w, o)); oracle_errors.append(metrics(oracle[:-1], oracle[-1]))
        residual = abs(y - X @ w - o)
        gaps.append((residual[bad].min(), residual[keep].max()))
    errors, oracle_errors, gaps = map(np.array, (errors, oracle_errors, gaps))
    for k in (6, 10):
        sel = np.arange(240) % 4 == (6, 10).index(k)*2+1
        print(case_id, k, 'spikes: max weight/offset/field errors', errors[sel].max(axis=0), 'oracle max', oracle_errors[sel].max(axis=0), 'max paired excess', (errors[sel]-oracle_errors[sel]).max(axis=0), 'min spike/max clean residual', gaps[sel,0].min(), gaps[sel,1].max())
    failures = np.flatnonzero((gaps[:, 0] < .1) | (gaps[:, 1] > .03))
    print('case', case_id, 'residual separation failures', len(failures), 'first trial indices', failures[:12])
