import importlib.util
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'alpha_m97_c02_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_alpha_m97_c02_s00.pkl').open('rb') as stream:
    d = pickle.load(stream)[0]['dataset']
tr = d['train']
lab = tr['lab_id']
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
n = len(lab)
p, u, f, s0 = m._prepare(d)
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - s0[:n, None]
r = tr['instant_rate_obs']
results = {}
for complete_only in (False, True):
    ec, er = np.empty_like(y), np.empty_like(r)
    taus = []
    for c in np.unique(lab):
        keep = lab != c
        if complete_only:
            keep &= ~missing
        a = np.flatnonzero(keep)
        b = np.flatnonzero(lab == c)
        lt, w = m._fit(p[a], u[a], f[a], dt, t, y[a], r[a])
        C, R, *_ = m._basis(lt, p[b], u[b], f[b], dt, t)
        ec[b], er[b] = y[b] - C @ w, r[b] - R @ w
        taus.append(np.exp(lt))
    results[complete_only] = ec, er
    print('complete-only training:', complete_only, 'fold tau span:', np.ptp(taus, axis=0))
    for name, sel in [('missing', missing), ('complete', ~missing)]:
        print(name, 'strain and rate median/p90:', np.quantile(abs(ec[sel]), [.5,.9]), np.quantile(abs(er[sel]), [.5,.9]))
    print('largest lab/horizon absolute strain median:', max(np.median(abs(ec[lab == c]),axis=0).max() for c in np.unique(lab)))
a, b = results[False][0], results[True][0]
print('missing coupon horizon-median absolute error decreased on', int(np.count_nonzero(np.median(abs(b[missing]), axis=1) < np.median(abs(a[missing]), axis=1))), 'of', missing.sum())

cal = d['calibration_load_cells']
base = m._calibrate(cal)
with TemporaryDirectory() as tmp:
    m._ARCHIVE = Path(tmp) / 'archive_calibration_history.json'
    m._ARCHIVE.write_text(json.dumps({'channels': {str(c): {'gain': float(g), 'bias': float(b)} for c, (g,b) in enumerate(base)}}))
    for count in (0, 1):
        mask = cal['lab_id'] != 2
        mask[np.flatnonzero(~mask)[:count]] = True
        np.testing.assert_allclose(m._calibrate({k: v[mask] for k, v in cal.items()}), base, rtol=0, atol=1e-12)
    m._ARCHIVE.write_text('{}invalid')
    np.testing.assert_allclose(m._calibrate(cal), base, rtol=0, atol=1e-12)
print('Sparse channel uses calibration archive; clean channels do not read it.')
