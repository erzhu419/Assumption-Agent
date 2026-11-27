import importlib.util
import itertools
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['dataset']
cal = d['calibration_load_cells']
L, R = cal['true_channel_loads'], cal['bridge_raw']
A = m._bridge_basis(L)
J = [j for k in (2, 3) for j in itertools.combinations_with_replacement(range(4), k)]
C = np.column_stack([np.ones(len(L)), L, *[np.prod(L[:, j], axis=1) for j in J]])
Q = np.column_stack([L[:, i] + s * L[:, j] for i, j in itertools.combinations(range(4), 2) for s in (-1, 1)])
# Prespecified splits: lower amplitudes to upper amplitudes, and mixed loads to single-axis sweeps.
amp = np.max(abs(L), axis=1)
for name, hold in [('high amplitude', amp >= np.quantile(amp, .75)), ('single-axis', np.count_nonzero(L, axis=1) == 1)]:
    train = ~hold
    b = m._fit_bridge(L[train], R[train])
    forward = A @ b
    e = np.linalg.norm(R - forward, axis=1)
    good = e < .002
    tr, te = train & good, hold & good
    assert te.any() and np.linalg.matrix_rank(A[tr]) == A.shape[1]
    inv = m._invert_bridge(R[te], b)
    print(name, 'clean/held', int(te.sum()), int(hold.sum()), 'clean load-sum RMSE/max', np.sqrt(np.mean((inv-L[te]).sum(1)**2)), np.max(abs((inv-L[te]).sum(1))))
    # Compare a dense cubic and tangent terms for denominators 1 + a*(load_i +/- load_j)^2.
    # Compete on held forward residuals; no alternative is selected on the held measurements.
    def rms(a):
        return np.sqrt(np.mean(np.sum(a*a, axis=1)))
    dense = C @ np.linalg.lstsq(C[tr], R[tr], rcond=None)[0]
    rat = np.column_stack([A, *(forward[:, k, None] * Q**2 for k in range(4))])
    tangent = rat @ np.linalg.lstsq(rat[tr], R[tr], rcond=1e-7)[0]
    print('  held forward RMS sparse/dense/rational-tangent', *(rms(v[te] - R[te]) for v in (forward, dense, tangent)), 'gross residuals', e[hold & ~good].round(3))

tr, te = d['train'], d['test']
x, _ = m._features(tr, te)
b = m._bridge(cal)
labs = tr['lab_id']
for lab in np.unique(labs):
    keep = labs != lab
    fit = {k:v[keep] for k,v in tr.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(labs)}
    taus, theta = m._material(fit, b, x[keep], d['past_dt'], d['eval_times'])
    past, hold = m._loads(tr, b)
    curve, rate = m._basis(taus, past[~keep], hold[~keep], x[~keep], d['past_dt'], d['eval_times'])
    ec = tr['strain_curves_obs'][~keep] - tr['strain_0'][~keep,None] - curve @ theta
    er = tr['instant_rate_obs'][~keep] - rate @ theta
    print('held lab', lab, 'signed curve/rate medians and rate median absolute error', np.median(ec), np.median(er), np.median(abs(er)))
