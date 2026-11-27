import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
a = m._bridge_basis(x)
records = {'current': [], 'sparse nonlinear': []}
for va in np.array_split(np.random.default_rng(704).permutation(len(x)), 5):
    tr = np.setdiff1d(np.arange(len(x)), va)
    b = m._fit_bridge(x[tr], y[tr])
    clean = tr[np.linalg.norm(a[tr] @ b - y[tr], axis=1) < .0015]
    keep = va[np.linalg.norm(a[va] @ b - y[va], axis=1) < .0015]
    assert len(keep) and len(clean) > a.shape[1]
    sparse = np.zeros_like(b)
    for j in range(y.shape[1]):
        support = np.r_[np.ones(5, dtype=bool), np.abs(b[5:, j]) > .01]
        sparse[support, j] = np.linalg.lstsq(a[clean][:, support], y[clean, j], rcond=None)[0]
    high = np.linalg.norm(x[keep], axis=1) >= np.quantile(np.linalg.norm(x[tr], axis=1), .8)
    for label, coef in [('current', b), ('sparse nonlinear', sparse)]:
        err = np.abs((m._invert_bridge(y[keep], coef) - x[keep]).sum(axis=1))
        records[label].append((err, err[high]))
for label, rows in records.items():
    all_err = np.concatenate([e for e, _ in rows]); high_err = np.concatenate([e for _, e in rows])
    print(label, 'worst fold median / max total-load error:', max(np.median(e) for e, _ in rows), all_err.max(), 'high-load count / max:', len(high_err), high_err.max())
