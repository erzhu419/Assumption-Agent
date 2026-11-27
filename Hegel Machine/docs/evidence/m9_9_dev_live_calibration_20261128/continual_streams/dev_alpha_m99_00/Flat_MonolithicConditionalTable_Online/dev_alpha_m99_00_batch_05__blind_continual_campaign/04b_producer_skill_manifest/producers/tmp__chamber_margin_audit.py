import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, v = item['dataset'], item['campaign_diagnostics']
out = m.solve_batch(d)
tau, th = out['estimated_taus'], out['hereditary_weights']
assert .05 < tau[0] < tau[1] < 25
bridge = m._bridge(d['calibration_load_cells'])
x = m._features(v)
t = d['eval_times']
past, h, _ = m._coupon(v, x, bridge)
s = x @ th[3:] + th[2]
y = v['strain_curves_obs'] - v['strain_0'][:, None]
r = v['instant_rate_obs']
lab = v['lab_id']
complete = np.isfinite(v['coupon_features_raw']).all(axis=1)
curve, rate = m._deployment(h, x, tau, th, t)
z = np.zeros((len(h), 2))
for j, dt in enumerate(d['past_dt']):
    z += -np.expm1(-dt / tau) * (past[:, j, None] - z)
load_c = h[:, None] * (-np.expm1(-t[:, None] / tau) @ th[:2])
load_r = h * np.sum(th[:2] / tau)

def fit_predict(kind, keep):
    p0, low, high = [0.25, 0.6, 7.5], [0., .03, .6], [1., 5., 50.]
    if kind != 'base':
        p0 += [0.]; low += [0. if kind == 'quotient' else -1.]; high += [1.]
    def predict(p):
        w, a, b = p[:3]
        f = -w*a*np.expm1(-t/a) - (1-w)*b*np.expm1(-t/b)
        ss = s.copy()
        if kind == 'quotient': ss = ss / (1 + p[3]*s*s)
        if kind == 'pair': ss = ss + p[3]*(x[:,0]-x[:,1])
        if kind == 'interaction': ss = ss + p[3]*h*s
        cc = load_c + ss[:,None]*f
        rr = load_r + ss
        if kind == 'memory':
            cc -= p[3] * ((z*th[:2]) @ -np.expm1(-t[:,None]/tau).T)
            rr -= p[3] * (z @ (th[:2]/tau))
        return cc, rr
    def residual(p):
        cc, rr = predict(p)
        return np.r_[(cc[keep]-y[keep]).ravel(), np.sqrt(len(t))*(rr[keep]-r[keep])]
    p = least_squares(residual, p0, bounds=(low,high), loss='soft_l1', f_scale=.003).x
    return predict(p)

for kind in ['base','memory','quotient','pair','interaction']:
    stats = []
    for l in np.unique(lab):
        fit = complete & (lab != l)
        held = complete & (lab == l)
        cc, rr = fit_predict(kind, fit)
        e = abs(cc[held]-y[held]); q = abs(rr[held]-r[held])
        stats.append([e.mean(), q.mean(), e[:,:4].mean(), e[:,-4:].mean()])
    print(kind, 'worst lab MAE [strain, rate, early, late]', np.max(stats, axis=0).round(7))
for mask, name in [(complete,'complete'), (~complete,'imputed')]:
    print(name, 'rows', mask.sum(), 'fixed-candidate strain/rate MAE', np.mean(abs(curve[mask]-y[mask])), np.mean(abs(rate[mask]-r[mask])))
zero, _ = m._deployment(h, x, tau, th, np.array([0.]))
eps = 1e-6
near, _ = m._deployment(h, x, tau, th, np.array([eps]))
assert np.max(abs(zero)) < 1e-12
err = np.max(abs((near[:,0]-zero[:,0])/eps-rate))
assert err < 1e-4
print('zero-strain and initial derivative verified; max finite-difference error', err)
