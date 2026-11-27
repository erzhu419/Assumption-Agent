import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c03_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(root + 'public_alpha_m97_c03_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
g, b = s._calibrate(d['calibration_load_cells'])
x, xt = s._align(tr, te)
assert d['shear_banding_index'] <= .50
out = s.solve_alpha_m97_c03_s00(d)
tau, w = out['estimated_taus'], out['hereditary_weights']
for split, xa in ((tr, x), (te, xt)):
    lab = split['lab_id']
    hist = g[lab, None] * split['past_stress_cmd'] + b[lab, None]
    hold = g[lab] * split['stress_hold_cmd'] + b[lab]
    assert np.max(np.abs(hist)) <= d['crit_stress']
    z = np.zeros((len(lab), 2))
    for stress, step in zip(hist.T, dt):
        decay = np.exp(-step/tau)
        z = decay*z + (1-decay)*stress[:, None]
    gap = hold[:, None]-z
    drift = w[2] + xa @ w[3:]
    c, r = s._design(split, xa, g, b, dt, t)(np.log(tau))
    np.testing.assert_allclose(c @ w, (gap*w[:2]) @ (-np.expm1(-t[:, None]/tau)).T + drift[:, None]*t, atol=1e-12)
    np.testing.assert_allclose(r @ w, gap @ (w[:2]/tau) + drift, atol=1e-12)
print('Both splits satisfy the linear regime; branch recurrence, strain envelope, and initial slope agree with their independent formulas.')

lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
for c in np.unique(lab):
    held = lab == c
    def design(mask):
        return s._design({k: tr[k][mask] for k in ('lab_id','past_stress_cmd','stress_hold_cmd')}, x[mask], g, b, dt, t)
    lt, beta = s._fit(design(~held), y[~held], r[~held], dt, t)
    a, z = design(held)(lt)
    ec, er = y[held]-a@beta, r[held]-z@beta
    for ss in np.unique(tr['stratum'][held]):
        m = complete[held] & (tr['stratum'][held] == ss)
        if m.any():
            print(f'lab={c} stratum={ss} complete n={m.sum()} signed medians strain/rate={np.nanmedian(ec[m]):+.5f}/{np.nanmedian(er[m]):+.5f} absolute P80={np.nanquantile(np.abs(ec[m]), .8):.5f}/{np.nanquantile(np.abs(er[m]), .8):.5f}')
