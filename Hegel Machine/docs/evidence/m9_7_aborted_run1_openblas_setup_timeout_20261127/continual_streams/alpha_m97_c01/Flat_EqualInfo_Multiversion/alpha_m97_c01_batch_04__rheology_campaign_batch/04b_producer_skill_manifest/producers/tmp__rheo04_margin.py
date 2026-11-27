import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p + 'alpha_m97_c01_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open(p + 'public_alpha_m97_c01_batch_04.pkl', 'rb'))[0]
d, gt = item['dataset'], item['gt']
tr = d['train']
lab = tr['lab_id']
n = len(lab)
dt, t = d['past_dt'], d['eval_times']
state = tuple(v[:n] for v in m._prepare(d))
x = state[2]
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
# Test omitted grammar on complete rows so feature error cannot masquerade as a drift term.
u = np.column_stack([x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)])
v = np.column_stack([x[:, i]*x[:, j]*x[:, k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)])
q = np.column_stack([x[:, i]/(1 + (x[:, i]+sign*x[:, j])**2) for i in range(4) for j in range(i+1, 4) for sign in (-1, 1)])
flow = np.log1p(t)
variants = {'prescribed': np.empty((n,0)), 'pairwise_and_three_way': np.column_stack((u,v)), 'rational_pairwise': q}
for name, extra in variants.items():
    scores = []
    for c in np.unique(lab):
        fit, test = (lab != c) & complete, (lab == c) & complete
        tau, _ = m._fit(tuple(z[fit] for z in state), dt, t, y[fit], r[fit])
        a,b = m._design(tau, state, dt, t)
        a = np.concatenate((a, extra[:,None,:]*flow[None,:,None]), axis=2)
        b = np.column_stack((b, extra))
        ac, br, yc, yr = a[fit], b[fit], y[fit], r[fit]
        cm, rm = np.isfinite(yc), np.isfinite(yr)
        design = np.concatenate((ac[cm], br[rm]))
        obs = np.r_[yc[cm], yr[rm]]
        wc = np.r_[np.ones(cm.sum()), np.full(rm.sum(), t[0])]
        for _ in range(9):
            coef = np.linalg.lstsq(design*wc[:,None], obs*wc, rcond=None)[0]
            res = obs - design@coef
            ec = np.full_like(yc, np.nan); ec[cm] = res[:cm.sum()]
            sc = np.maximum(m._mad(ec, axis=0), max(.35*m._mad(ec), 1e-7))
            sr = max(m._mad(res[cm.sum():]), 1e-7)
            sigma = np.r_[np.broadcast_to(sc, yc.shape)[cm], np.full(rm.sum(), sr)]
            wc = 1/np.hypot(sigma, res/2.38)
        e, f = a[test]@coef-y[test], b[test]@coef-r[test]
        med = np.nanmedian(abs(e[:,-3:]),axis=0).max()
        scores.append([med, np.sqrt(np.nanmean(f*f)), np.nanmedian(f)])
    print(name, 'per-lab [worst late horizon median abs, rate RMSE, signed median rate residual]', np.round(scores,6).tolist())
out = m.solve_alpha_m97_c01_batch_04(d)
cal = m._calibrate(d['calibration_load_cells'])
print('per-channel gain/bias absolute error', np.round(abs(cal - np.column_stack((gt['stress_gains'],gt['stress_biases']))),8))
print('relative tau error', np.round(abs(out['estimated_taus']/gt['taus']-1),6))
print('worst hereditary/mode absolute errors', np.max(abs(out['hereditary_weights']-gt['theta'])), np.max(abs(out['regime_mode_weights']-gt['vp_theta'])))
