import contextlib, copy, io
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/b03_validate.py').read())

# All candidate corrections are trained on other labs, then scored separately
# on the held-out trajectories and initial rates.
gap = np.broadcast_to(h[:n, None], (n, 2)).copy()
state = np.zeros_like(gap)
for j, dt in enumerate(d['past_dt']):
    w = np.exp(-dt / tau)
    state = w * state + (1 - w) * p[:n, j, None]
gap -= state
stress = h[:n]
X = x[:n]
phi = times / (1 + 0.85 * times)
flow = np.r_[phi, 1.]
contrast = np.r_[-np.expm1(-times / tau[0]), 1 / tau[0]]
missing = np.isnan(a['coupon_features_raw']).any(axis=1)
material = np.column_stack([X[:, i] * X[:, j] for i in range(4) for j in range(i, 4)] + [X[:, i] * X[:, j] * X[:, k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)])
corrections = {
    'prescribed': np.empty((*y.shape, 0)),
    'nonlinear material pairs/triples': material[:, None, :] * flow[None, :, None],
    'extra rational stress slip': np.column_stack([stress / (1 + stress**2), np.sign(stress)*np.sqrt(abs(stress))/(1+stress**2)])[:, None, :] * flow[None, :, None],
    'second contrast kernel': (gap[:, 0] - gap[:, 1])[:, None, None] * np.r_[-np.expm1(-times / tau[1]), 1 / tau[1]][None, :, None],
    'missingness diagnostic': missing[:, None, None] * flow[None, :, None],
}
for name, A in corrections.items():
    metrics = []
    for lab in np.unique(a['lab_id']):
        held = a['lab_id'] == lab
        mask = np.broadcast_to(~held[:, None], y.shape)
        t, b = s._fit(m, y, mask)
        err = m.design(t) @ b - y
        if A.shape[-1]:
            keep = mask & np.isfinite(err)
            scales = np.nanmedian(np.abs(np.where(mask, err, np.nan)), axis=0) * 1.4826
            w = np.broadcast_to(np.maximum(scales, 1e-6), y.shape)[keep]
            z = least_squares(lambda coeff: (A[keep] @ coeff + err[keep])/w,
                              np.zeros(A.shape[-1]), loss='cauchy', f_scale=2.5).x
            err = err + A @ z
        e = err[held]
        metrics.append((np.nanmedian(abs(e[:,:-1])), rmse(e[:,-1]), np.nanmedian(e[:,-1])))
    scores = np.asarray(metrics)
    print(name, 'worst median strain / rate RMSE', np.round(scores[:,:2].max(axis=0), 6), 'rate bias by lab', np.round(scores[:,2], 6))

base = s._calibrate(d)
worst = {}
sweep = d['calibration_load_cells']
for lab in range(4):
    rows = np.flatnonzero(sweep['lab_id'] == lab)
    for count in (2, 3, 4, 5, 10):
        shifts = (0.,) if count == 2 else (0., -3.01, 3.01, -5., 5.)
        for amount in shifts:
            for position in (range(count) if amount else (0,)):
                c = {k:v.copy() for k,v in sweep.items()}
                c['true_stress'][rows[position]] += amount
                keep = np.ones(len(c['lab_id']), dtype=bool)
                keep[rows[count:]] = False
                c = {k:v[keep] for k,v in c.items()}
                err = float(np.max(abs(s._calibrate(dict(d, calibration_load_cells=c))[lab]-base[lab])))
                key = (count, abs(amount))
                if key not in worst or err > worst[key][0]:
                    worst[key] = (err, lab, position, amount)
for key, v in sorted(worst.items()):
    print('calibration count / spike magnitude', key, 'worst error / lab / position / signed spike', tuple(round(c,7) for c in v))
