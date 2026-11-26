import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_w02_s00_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open('/holdout_task/public_alpha_w02_s00.pkl', 'rb') as f:
    records = pickle.load(f)

def rmse(x):
    return np.sqrt(np.mean(x*x))

for j, record in enumerate(records):
    d, gt = record['dataset'], record['gt']
    tr, te, t = d['train'], d['test'], d['eval_times']
    pred = solver.solve_alpha_w02_s00(d)
    tau, theta = pred['estimated_taus'], pred['hereditary_weights']
    assert .2 <= tau[0] < tau[1] <= 8 and np.array_equal(pred['viscoplastic_weights'], np.zeros(2))
    err = pred['predicted_strain']-gt['te_clean']
    rate = pred['initial_rate_pred']-gt['te_true_rate0']
    print('record', j, 'tau', tau, 'max relative tau error', max(abs(tau/gt['taus']-1)), 'max absolute theta error', max(abs(theta-gt['theta'])))
    complete = np.isfinite(te['coupon_features_raw']).all(axis=1)
    for lab in np.unique(te['lab_id']):
        for present in (True, False):
            q = (te['lab_id']==lab)&(complete==present)
            if not q.any(): continue
            # A static coupon drift error contributes rate_error * t exactly;
            # deviations isolate the stress-history and transient branches.
            transient = err[q]-rate[q,None]*t
            print('lab',lab,'complete',present,'n',q.sum(),'early/late/initial/transient RMSE',*(round(v,6) for v in [rmse(err[q,:3]),rmse(err[q,-3:]),rmse(rate[q]),rmse(transient)]))
    for lab in np.unique(tr['lab_id']):
        q = tr['lab_id']==lab
        fold = dict(d, train={k:v[~q] for k,v in tr.items()}, test={k:np.concatenate((v[q],te[k]),axis=0) if k in te else v[q] for k,v in tr.items()})
        fold_out = solver.solve_alpha_w02_s00(fold)
        n = q.sum()
        present = np.isfinite(tr['coupon_features_raw'][q]).all(axis=1)
        e = fold_out['predicted_strain'][:n]-tr['strain_curves_obs'][q]
        r = fold_out['initial_rate_pred'][:n]-tr['instant_rate_obs'][q]
        print('held lab',lab,'tau',fold_out['estimated_taus'],'complete onset p90',np.nanquantile(abs(r[present]),.90),'complete early/late median absolute residual',np.nanmedian(abs(e[present,:3])),np.nanmedian(abs(e[present,-3:])))
