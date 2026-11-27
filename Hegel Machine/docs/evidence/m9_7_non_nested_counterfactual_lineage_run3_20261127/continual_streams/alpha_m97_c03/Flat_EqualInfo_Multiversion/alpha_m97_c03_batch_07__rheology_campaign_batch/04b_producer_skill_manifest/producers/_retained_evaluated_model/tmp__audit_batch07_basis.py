import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch07', root + 'alpha_m97_c03_batch_07_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c03_batch_07.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
s = d['train']; lab = s['lab_id']; n = len(lab)
past, hold, x, act, offset, dt, t = m._prepare(d)
past, hold, x, act = (a[:n] for a in (past, hold, x, act))
y = s['strain_curves_obs'] - offset[:n, None]
r = s['instant_rate_obs']
complete = np.isfinite(s['coupon_features_raw']).all(1)
D = np.maximum(0, np.abs(past) - d['crit_stress'])**2 @ dt
seen = {'inactive': D == 0, 'weak': (D > 0) & (D < 2), 'strong': D >= 2}
print('Complete-coupon damage counts:', {k: int((v & complete).sum()) for k, v in seen.items()})
errors = {k: (np.zeros_like(y), np.zeros_like(r)) for k in ('saved', 'refit', 'gap_quotient', 'damage_quotient')}
for j in np.unique(lab):
    h = lab == j; f = ~h & complete
    p, w = m._fit((past[f], hold[f], x[f], act[f], dt, t), y[f], r[f])
    c, b = m._design(p, past, hold, x, act, dt, t)
    errors['saved'][0][h] = c[h] @ w - y[h]
    errors['saved'][1][h] = b[h] @ w - r[h]
    tau = np.exp(p)
    z = np.zeros((n, 2))
    for i, delta in enumerate(dt):
        a = np.exp(-delta / tau)
        z = a*z + (1-a)*past[:, i, None]
    gap = hold[:, None] - z
    q = gap[:, 0] - gap[:, 1]
    # Alternative bases retain zero strain increment and the exact derivative at t=0.
    exp_basis = -np.expm1(-t/tau[0]); log_basis = np.log1p(t)
    sat = D / np.sqrt(1+D**2) - act
    extras = {
        'refit': (c, b),
        'gap_quotient': (np.concatenate((c, (q/(1+q*q))[:, None, None]*exp_basis[None, :, None]), axis=2), np.column_stack((b, q/(1+q*q)/tau[0]))),
        'damage_quotient': (np.concatenate((c, (sat*hold)[:, None, None]*exp_basis[None, :, None], (sat*z[:, 0])[:, None, None]*log_basis[None, :, None]), axis=2), np.column_stack((b, sat*hold/tau[0], sat*z[:, 0])))
    }
    e, er = c[f] @ w - y[f], b[f] @ w - r[f]
    nc = e.size
    target = np.r_[y[f].ravel(), r[f]]
    for name, (ac, ar) in extras.items():
        a = np.vstack((ac[f].reshape(nc, -1), ar[f]))
        err = np.r_[e.ravel(), er]
        for _ in range(5):
            weight = np.empty(len(err))
            for sl in (slice(None, nc), slice(nc, None)):
                mid = np.median(err[sl]); scale = max(1e-7, 1.4826*np.median(abs(err[sl]-mid)))
                weight[sl] = np.sqrt(np.minimum(1., 2.5*scale/np.maximum(1e-12, abs(err[sl]-mid)))) / scale
            coef = np.linalg.lstsq(a*weight[:,None], target*weight, rcond=1e-10)[0]
            err = a @ coef - target
        errors[name][0][h] = ac[h] @ coef - y[h]
        errors[name][1][h] = ar[h] @ coef - r[h]
for name, (e, er) in errors.items():
    def metrics(mask):
        return tuple(round(float(v), 5) for v in (np.median(abs(e[mask])), np.quantile(abs(e[mask]), .9), np.median(abs(er[mask])), np.quantile(abs(er[mask]), .9)))
    print(name, 'curve median/p90, rate median/p90 by complete lab:', {int(j): metrics((lab == j) & complete) for j in np.unique(lab)})
    print(name, 'curve median/p90, rate median/p90 by complete damage slice:', {k: metrics(v & complete) for k,v in seen.items() if (v & complete).sum() >= 5})
