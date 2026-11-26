import os
import runpy
import time
import numpy as np

v = runpy.run_path('/tmp/eq02_verify.py')
m, verify, fixture = (v[k] for k in ('mod', 'verify', 'fixture'))
for p in ('/holdout_task/public_eq02_studies.pkl', '/holdout_task/validate_public_schema.py'):
    print(p, 'available' if os.path.isfile(p) else 'not mounted', flush=True)

# Hold out a center from calibration, but keep its own reference controls for alignment.
worst = [0., 0., 0., 0.]
for seed in range(876, 881):
    st, truth, theta = fixture(seed, 19, 1.3, .63)
    panel = st['panel']; center = panel['center_id']; s = panel['stratum']
    aligned, _ = m._panel(panel)
    # Independently compare the imputation with an equally weighted sum of cells.
    observed = np.isfinite(panel['X_panel_raw'])
    for j in range(aligned.shape[1]):
        numer = denom = 0.
        for c in np.unique(center):
            for k in range(3):
                idx = (center == c) & (s == k); valid = idx & observed[:, j]
                numer += aligned[valid, j].sum()/idx.sum(); denom += valid.sum()/idx.sum()
        np.testing.assert_allclose(aligned[~observed[:, j], j], numer/denom, atol=1e-12)
    optics = st['interferometry']
    phase = (np.angle(optics['signal_phasor']) - np.angle(optics['ref_phasor']) + np.pi) % (2*np.pi) - np.pi
    H = np.column_stack([np.ones(len(s)), s == 1, s == 2, aligned, phase]); y = st['y_assay_raw']; cal = st['calibration_indices']; target = st['stage3_indices']
    for c in np.unique(center):
        train = cal[center[cal] != c]; test = target[center[target] == c]; loc = center[target] == c
        robust = m._cauchy(H[train], y[train]); ols = np.linalg.lstsq(H[train], y[train], rcond=None)[0]
        sparse = np.zeros(19); sparse[:5] = .63*np.array([1,-.9,1,-.75,.85]); expected = st['X_cand'][loc]@sparse
        rr = y[test] - H[test]@robust - expected; ro = y[test] - H[test]@ols - expected
        result = [np.mean(abs(aligned[center == c] - truth[center == c])), np.linalg.norm(robust-theta)/np.linalg.norm(theta), np.sqrt(np.mean(rr**2)), np.sqrt(np.mean(ro**2))]
        worst = np.maximum(worst, result)
print('leave-one-center-out maxima: panel MAE, Cauchy relative theta error, Cauchy residual RMSE, OLS residual RMSE:', worst, flush=True)
assert worst[0] <= .11 and worst[1] <= .14

# Similar-strength columns near the support switching penalty stress proof compression.
for p, seed in [(18, 5216), (19, 5217)]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(68,p)) + .55*rng.normal(size=(68,8)) @ rng.normal(size=(8,p)); X /= np.linalg.norm(X, axis=0)
    y = X @ (.42*rng.choice([-1.,1.], p)) + .10*rng.normal(size=len(X)); l2 = .035; l0 = .085; M = .55
    start = time.monotonic(); cert = m._box_l0(X,y,l2,l0,M); elapsed = time.monotonic()-start
    _, gap, leaves = verify(X,y,l2,l0,M,cert)
    print(f'connected near-switch p={p}: {elapsed:.3f}s, gap={gap:.3g}, leaves={leaves}', flush=True)
    assert -1e-10 <= gap <= 1e-5 and leaves <= 2048 and elapsed <= 9
