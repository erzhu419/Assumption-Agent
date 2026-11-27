import importlib.util, pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'alpha_m97_c02_batch_01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_alpha_m97_c02_batch_01.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
dt, t, (b, _) = m._prepare(d)
s = d['train']
y = s['strain_curves_obs'] - s['strain_0'][:, None]
r = s['instant_rate_obs']
lab = s['lab_id']
complete = np.isfinite(s['coupon_features_raw']).all(axis=1)
active = b[3] > 0
sub = lambda b, mask: tuple(x[mask] for x in b)
rmse = lambda x: np.sqrt(np.mean(x*x))
print('dataset fields', sorted(d), 'train counts by lab (complete, incomplete)', [(int(np.sum((lab == k)&complete)), int(np.sum((lab == k)&~complete))) for k in range(4)])
for method in ('baseline', 'complete-fit', 'nonlinear-drift-probe'):
    ec = np.empty_like(y)
    er = np.empty_like(r)
    folds = []
    for k in range(4):
        te = lab == k
        tr = (lab != k) & complete if method == 'complete-fit' else lab != k
        tau, w = m._fit(dt, t, sub(b, tr), y[tr], r[tr])
        folds.append(tau)
        c, a = m._basis(tau, dt, t, sub(b, te))
        ec[te], er[te] = c @ w - y[te], a @ w - r[te]
        if method == 'nonlinear-drift-probe':
            # A drift correction must act through the same time law and its initial slope.
            # Fit the prespecified directions only on the outer training labs.
            i, j = b[2][:, 1], b[2][:, 2]
            z = np.column_stack(((i-j)**2, (i+j)**2, i/(1+j*j)-i))
            tr = (lab != k) & complete
            ct, at = m._basis(tau, dt, t, sub(b, tr))
            x = z[tr]
            offset = np.median((y[tr] - ct @ w)/np.log1p(t), axis=1)
            slope = r[tr] - at @ w
            beta = np.linalg.lstsq(x, (np.clip(offset, -.08, .08) + np.clip(slope, -.08, .08))/2, rcond=None)[0]
            delta = z[te] @ beta
            ec[te] += delta[:, None]*np.log1p(t)
            er[te] += delta
    groups = [(lab==k)&(active==on) for k in range(4) for on in (False, True)]
    print(method, 'worst complete terminal median, curve P90, rate RMSE, raw curve RMSE', np.round([max(np.median(abs(ec[g&complete,-1])) for g in groups), max(np.quantile(abs(ec[g]), .9) for g in groups), max(rmse(er[g]) for g in groups), rmse(ec)], 6), 'tau fold span', np.round(np.ptp(folds, axis=0), 6), 'complete signed rate medians by lab', np.round([np.median(er[(lab==k)&complete]) for k in range(4)], 6))

# Check that one transient in a real sweep does not move a well-supported in-batch fit.
cal = d['calibration_load_cells']
x, z, channel = (np.asarray(cal[key]) for key in ('cmd_stress', 'true_stress', 'lab_id'))
base = m._calibration(d)
worst = 0.
for k in range(4):
    mask = channel == k
    for row in np.flatnonzero(mask):
        for spike in (-3.1, 3.1):
            noisy = z.copy(); noisy[row] += spike
            got = m._calibration(dict(d, calibration_load_cells=dict(cal, true_stress=noisy)))
            delta = (got[k,0]-base[k,0])*x[mask]+got[k,1]-base[k,1]
            worst = max(worst, float(np.max(abs(delta))))
print('max calibrated stress change after one +/-3.1 transient', worst)
assert worst < .001
