import importlib.util
import pickle
import numpy as np

path = '/holdout_task/alpha_w04_batch_08_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_w04_batch_08.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
a, test = d['train'], d['test']
dt, t, sy, panel = (d[k] for k in ('past_dt', 'eval_times', 'yield_stress', 'calibration_load_cells'))
lab = a['lab_id']
complete = np.isfinite(a['coupon_features_raw']).all(axis=1)

def subset(i):
    return {k:v[i] for k,v in a.items()}

x, z = s._align(a, test)
tau, aff, w = s._fit_batch(a, x, dt, t, sy, panel)
print('fit taus', tau.round(6), 'affines', aff.round(6).tolist(), 'weights', w.round(6))
for name, split in [('train', a), ('test', test)]:
    h, q = s._stress(split, aff)
    print(name, 'history and hold above yield by lab:', [(int(np.sum((split['lab_id']==j) & (np.max(abs(h),axis=1)>sy))), int(np.sum((split['lab_id']==j) & (abs(q)>sy)))) for j in range(4)])

# Independently check the rate handoff on the actual stress histories.
h, q = s._stress(a, aff)
zero, rate = s._design(tau, dt, np.array([0.]), h, q, x, sy)
step = 1e-5
forward, _ = s._design(tau, dt, np.array([step, 2*step]), h, q, x, sy)
fd = (4*forward[:,0]-forward[:,1]-3*zero[:,0])/(2*step)
np.testing.assert_array_equal(zero, 0.)
np.testing.assert_allclose(fd, rate, atol=1e-7, rtol=1e-7)
print('t=0 strain/rate handoff max coefficient error', np.max(abs(fd-rate)))

# Compare the production fit with complete-coupon estimation of the shared law.
# Both variants predict without using the validation rate for this comparison.
rng = np.random.default_rng(613)
folds = [[] for _ in range(4)]
for j in range(4):
    for c in (True, False):
        for fold, rows in zip(folds, np.array_split(rng.permutation(np.flatnonzero((lab==j)&(complete==c))),4)):
            fold.extend(rows)
errors = {key:(np.empty_like(a['strain_curves_obs']), np.empty_like(a['instant_rate_obs'])) for key in ('all','complete_fit')}
for fold in folds:
    j = np.asarray(fold); i = np.setdiff1d(np.arange(len(lab)),j)
    tr, te = subset(i), subset(j)
    x, z = s._align(tr, te)
    for name, use in [('all', np.ones(len(i), dtype=bool)), ('complete_fit', complete[i])]:
        model = s._fit_batch(subset(i[use]), x[use], dt, t, sy, panel)
        p, r = s._predict(tr,te,x,z,dt,t,sy,model,correction='none')
        errors[name][0][j] = p - te['strain_curves_obs']; errors[name][1][j] = r - te['instant_rate_obs']
for name, (ce, re) in errors.items():
    med = []; bias = []
    for j in range(4):
        use = (lab==j)&complete
        med.append([np.median(abs(ce[use])), np.median(abs(re[use]))])
        bias.append([np.median(ce[use,-1]), np.median(re[use])])
    print(name, 'complete coupon OOF per-lab strain/rate MedAE', np.round(med,6).tolist(), 'signed late strain/rate bias', np.round(bias,6).tolist())
