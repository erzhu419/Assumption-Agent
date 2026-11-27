import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c02_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(root + 'public_alpha_m97_c02_batch_07.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
a = d['train']
x, _ = s._aligned(a, d['test'])
dt, t = d['past_dt'], d['eval_times']
lab = a['lab_id']
complete = np.isfinite(a['coupon_features_raw']).all(axis=1)
v = s._inputs(a, x, s._calibration(d), dt, d['crit_stress'], d['shear_banding_index'])
y = a['strain_curves_obs'] - a['strain_0'][:, None]
r = a['instant_rate_obs']
for stop in (len(t), len(t)-1):
    ec, er = np.empty_like(y), np.empty_like(r)
    fits = []
    for c in np.unique(lab):
        tr, te = (lab != c) & complete, lab == c
        tau, w = s._fit(dt, t[:stop], tuple(q[tr] for q in v), y[tr, :stop], r[tr])
        C, R = s._design(tau, dt, t, tuple(q[te] for q in v))
        ec[te], er[te] = C @ w - y[te], R @ w - r[te]
        fits.append(np.r_[tau, w])
    print('training horizons', stop, 'tau ranges', np.round(np.array(fits)[:, :2].min(0), 6), np.round(np.array(fits)[:, :2].max(0), 6))
    print('lab/damage slice: n, earlier strain p90, final signed median, final p90, rate p90')
    for c in np.unique(lab):
        for act in (False, True):
            m = complete & (lab == c) & ((v[-1] > 0) == act)
            if m.any():
                print(int(c), act, int(m.sum()), np.round([np.quantile(abs(ec[m, :-1]), .9), np.median(ec[m, -1]), np.quantile(abs(ec[m, -1]), .9), np.quantile(abs(er[m]), .9)], 6).tolist())
    for tag, m in [('complete', complete), ('missing', ~complete)]:
        print(tag, 'final residual > 0.10, negative/positive', int(np.sum(ec[m, -1] < -.10)), int(np.sum(ec[m, -1] > .10)), 'of', int(m.sum()))
