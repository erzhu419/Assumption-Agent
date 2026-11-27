import importlib.util
import pickle
from itertools import combinations
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', p + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(p + 'public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']; x, y = c['ref_rosette'], c['raw_rosette']
F = m._rosette_features(x)
coef = m._fit_rosette(x, y)
clean = np.max(abs(F @ coef - y), axis=1) < 1e-3
S, U = m._forward(s)
obs = np.asarray(s['y_obs_raw']).reshape(-1, 4)
Z = s['Z_nuisance']; I = np.eye(Z.shape[1]); W = np.eye(S.shape[1])
cap, budget = s['u_box'], s['total_budget']
zridge, wridge = s['nuisance_ridge'], s['source_ridge']
P = np.linalg.solve(Z.T @ Z + zridge * I, Z.T)
T = S - Z @ (P @ S)

def reconstruct(b):
    v = m._invert_rosette(obs, b).ravel(); v -= Z @ (P @ v)
    w, _ = m._box_simplex(T.T @ T + wridge * W, T.T @ v, cap, budget)
    return w, np.tensordot(w, U, axes=(0, 0))

w0, u0 = reconstruct(coef)
# Quadratic seeds of rational denominators: diagonal squares and both signs
# of paired channels. x_i^2 and x_i*x_j are already spanned by the seeds.
q = np.column_stack([x*x] + [(x[:, i] + a*x[:, j])**2 for i,j in combinations(range(4), 2) for a in (-1, 1)])
rows = np.arange(len(x)); errors = [np.zeros(len(x)) for _ in range(3)]; changes = []
for block in np.array_split(rows, 5):
    train = ~np.isin(rows, block)
    b = m._fit_rosette(x[train], y[train]); f = F[train]; z = y[train]
    keep = np.max(abs(f @ b - z), axis=1) < 1e-3
    # Fit an alternative sparse polynomial and an alternative rational term.
    # Choose the denominator separately for each bank using training only.
    sparse = np.zeros_like(b); alt = []; pred = F @ b
    for bank in range(4):
        active = abs(b[:, bank]) > 1e-3
        a = f[keep][:, active]; target = z[keep, bank]
        sparse[active, bank] = np.linalg.lstsq(a, target, rcond=None)[0]
        best = (np.linalg.norm(a @ sparse[active, bank] - target), F[:, active] @ sparse[active, bank], 0.)
        for col in q.T:
            for alpha in (0.001, 0.01, 0.05, 0.15, 0.4):
                basis = F[:, active] / (1 + alpha * col[:, None]); t = basis[train][keep]
                v = np.linalg.lstsq(t, target, rcond=None)[0]; e = np.linalg.norm(t @ v - target)
                if e < best[0]: best = (e, basis @ v, alpha)
        alt.append(best[1])
    for err, pred in zip(errors, (pred, F @ sparse, np.array(alt).T)):
        err[block] = np.max(abs(pred[block] - y[block]), axis=1)
    w, u = reconstruct(b)
    changes.append((np.linalg.norm(w-w0)/np.linalg.norm(w0), np.linalg.norm(u-u0)/np.linalg.norm(u0)))
for name, err in zip(('current', 'sparse cubic', 'rational probe'), errors):
    print(name, 'clean OOF median, p95, max', np.quantile(err[clean], [.5,.95,1]), 'gross outliers >1e-3', np.count_nonzero(err>1e-3))
print('Calibration block deletion worst relative weights and field changes:', np.max(changes, axis=0))
# Differential source identification: the sum constraint leaves only contrasts.
contrast = np.vstack((np.eye(S.shape[1]-1), -np.ones(S.shape[1]-1)))
contrast, _ = np.linalg.qr(contrast)
for k, block in enumerate(np.array_split(np.arange(len(S)), len(s['mu_shifts']))):
    keep = ~np.isin(np.arange(len(S)), block); a = S[keep]; z = Z[keep]
    pz = np.linalg.solve(z.T @ z + zridge * I, z.T)
    v = m._calibrate(s); a0 = a - z @ (pz @ a); v0 = v[keep] - z @ (pz @ v[keep]);
    w, _ = m._box_simplex(a0.T @ a0 + wridge * W, a0.T @ v0, cap, budget)
    g = pz @ (v[keep] - a @ w); err = S[block] @ w + Z[block] @ g - v[block]
    print('held shift', k, 'RMS', np.sqrt(np.mean(err**2)), 'contrast sigma_min raw/projected', np.linalg.svd(a @ contrast, compute_uv=False)[-1], np.linalg.svd(a0 @ contrast, compute_uv=False)[-1], 'relative weight change', np.linalg.norm(w-w0)/np.linalg.norm(w0))
