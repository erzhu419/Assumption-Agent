import pickle, runpy
import numpy as np
m = runpy.run_path('/holdout_task/batch_solver.py')
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']; x = c['true_channel_loads']; y = c['bridge_raw']
A = m['_bridge_basis'](x)
terms = [()] + m['_TERMS']
bad = np.zeros(len(x), dtype=bool)
err = {name: np.empty_like(y) for name in ('full cubic', 'sparse cubic', 'rational difference')}
# Compare extrapolation at the largest labeled load magnitude with interleaved folds.
outer = np.argsort(np.linalg.norm(x, axis=1))[-12:]
folds = [outer] + [np.setdiff1d(np.arange(i, len(x), 5), outer) for i in range(5)]
for held in folds:
    train = np.ones(len(x), dtype=bool); train[held] = False
    b = m['_calibration']({'true_channel_loads': x[train], 'bridge_raw': y[train]})
    err['full cubic'][held] = A[held] @ b - y[held]
    clean = train & (np.linalg.norm(A @ b - y, axis=1) < 0.01)
    bad[held] = np.linalg.norm(err['full cubic'][held], axis=1) > 1.
    for j in range(4):
        selected = (np.abs(b[:, j]) > 0.01); selected[0] = True
        a = A[:, selected]; z = y[:, j]
        p = np.linalg.lstsq(a[clean], z[clean], rcond=None)[0]
        err['sparse cubic'][held, j] = a[held] @ p - z[held]
        # A quotient with a pairwise difference requires mixed fifth-order terms
        # beyond the identified cubic. Compare the concrete coupled quotient.
        q = x[:, j] - x[:, (j + 1) % 4]
        q2 = q * q
        from scipy.optimize import least_squares
        def residual(v):
            return ((a[clean] @ v[:-1]) / (1 + v[-1] * q2[clean])) - z[clean]
        v = least_squares(residual, np.r_[p, 0.001], bounds=(np.r_[np.full(len(p), -np.inf), 0], np.inf), gtol=1e-12).x
        err['rational difference'][held, j] = (a[held] @ v[:-1]) / (1 + v[-1] * q2[held]) - z[held]
print('gross corruption excluded only from clean metrics:', np.flatnonzero(bad))
tail = np.isin(np.arange(len(x)), outer)
for name, e in err.items():
    for label, use in [('all', ~bad), ('high-load', tail & ~bad)]:
        a=e[use]; print(name, label, 'channel RMSE', np.sqrt(np.mean(a*a)), 'axial raw-sum RMSE', np.sqrt(np.mean(a.sum(1)**2)), 'worst row', np.linalg.norm(a,axis=1).max())
# Diagnose lab nuisance in physical units using response labels excluded from fit.
tr=d['train']; te=d['test']; labs=tr['lab_id']; t=d['eval_times']; dt=d['past_dt']
u=m['_coupon_inputs'](tr, m['_features'](tr,te)[:len(labs)], m['_calibration'](c))
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
for lab in np.unique(labs):
    h=labs==lab
    ts,w=m['_fit'](tuple(a[~h] for a in u),dt,t,y[~h],r[~h]); a,b=m['_design'](ts,tuple(a[h] for a in u),dt,t)
    re=b@w-r[h]; le=(a@w-y[h])[:,-3:]
    print('held lab',lab,'signed rate/late curve median',np.nanmedian(re),np.nanmedian(le),'absolute medians',np.nanmedian(abs(re)),np.nanmedian(abs(le)))
