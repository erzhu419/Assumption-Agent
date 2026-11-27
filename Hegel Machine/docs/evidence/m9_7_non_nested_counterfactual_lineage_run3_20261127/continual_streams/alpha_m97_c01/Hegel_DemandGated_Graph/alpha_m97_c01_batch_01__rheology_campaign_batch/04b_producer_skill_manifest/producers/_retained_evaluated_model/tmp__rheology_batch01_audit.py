import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_batch_01_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c01_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
q = d['train']
lab = q['lab_id']
n = len(lab)
a = tuple(v[:n] for v in s._prepare(d))
dt, t = d['past_dt'], d['eval_times']
y = q['strain_curves_obs'] - q['strain_0'][:, None]
r = q['instant_rate_obs']
print('Dataset fields:', sorted(d), '; train fields:', sorted(q))
out = s.solve_alpha_m97_c01_batch_01(d)
assert all(np.isfinite(v).all() for v in out.values())
assert (out['estimated_taus'] > 0).all()

# A sum or difference of features is already in the linear drift span.
# Challenge that span with pairwise products, a three-way product basis, and
# a smooth rational hold-stress response, using only training labs for fits.
x = a[2][:, 1:]
h = a[1]
z = np.column_stack([x[:, i]*x[:, j] for i in range(4) for j in range(i+1, 4)] +
                    [x[:, i]*x[:, j]*x[:, k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)] +
                    [h/(1+h*h), h*h/(1+h*h)])
phi = np.log1p(t)
results = {'base': [], 'expanded': []}
for group in np.unique(lab):
    test = lab == group
    tr = ~test
    logtau, w = s._fit(dt, t, tuple(v[tr] for v in a), y[tr], r[tr])
    c, b = s._basis(logtau, dt, t, a)
    offsets = (c @ w - y, b @ w - r)
    scale = [max(1e-7, 2.5*1.4826*np.median(np.abs(e[tr]-np.median(e[tr])))) for e in offsets]
    ce = np.concatenate((c, phi[None, :, None]*z[:, None, :]), axis=-1)
    be = np.column_stack((b, z))
    design = np.vstack((ce[tr].reshape(-1, be.shape[1])/scale[0], be[tr]/scale[1]))
    target = np.r_[y[tr].ravel()/scale[0], r[tr]/scale[1]]
    ww = least_squares(lambda p: design@p-target, np.r_[w, np.zeros(z.shape[1])],
                       jac=lambda p: design.copy(), loss='cauchy', max_nfev=60).x
    for mode, (e, er) in [('base', offsets), ('expanded', (ce@ww-y, be@ww-r))]:
        for gate in (False, True):
            mask = test & ((a[-1] > 0) == gate)
            results[mode].append((np.median(abs(e[mask])), np.median(abs(er[mask])),
                                  np.max(np.median(abs(e[mask]), axis=0)),
                                  np.max(abs(np.median(e[mask], axis=0)))))
    print('held-out lab', group, 'signed strain/rate median residuals',
          [round(float(np.median(e[test])), 5) for e in offsets])
for mode, rows in results.items():
    print(mode, 'worst slice: strain AE, rate AE, horizon AE, horizon bias', np.round(np.max(rows, axis=0), 6))
