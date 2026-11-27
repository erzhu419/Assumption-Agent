import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'alpha_m97_c03_batch_07_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c03_batch_07.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d, g = item['dataset'], item['gt']
    tr, te = d['train'], d['test']
    xt, xe = m._align(tr, te)
    cal = m._calibrate(d['calibration_load_cells'])
    dt, t = d['past_dt'], d['eval_times']
    args = cal, dt, t, d['crit_stress'], d['shear_banding_index']
    complete = np.isfinite(tr['coupon_features_raw']).all(1)
    missing = ~np.isfinite(te['coupon_features_raw']).all(1)
    lab = tr['lab_id']
    cy = tr['strain_curves_obs'] - tr['strain_0'][:, None]
    ry = tr['instant_rate_obs']

    def design(mask):
        subset = {k: tr[k][mask] for k in ('lab_id', 'past_stress_cmd', 'stress_hold_cmd')}
        return m._design(subset, xt[mask], *args)

    print('Incomplete training coupons by lab:', np.bincount(lab[~complete], minlength=4))
    for name, eligible in [('complete-only', complete), ('include-imputed', np.ones(len(lab), bool))]:
        folds = []
        for held in np.unique(lab):
            fit, val = eligible & (lab != held), complete & (lab == held)
            taus, weights = m._fit(design(fit), cy[fit], ry[fit], dt, t)
            C, R = design(val)(taus)
            folds.append([np.nanmedian(abs(C @ weights - cy[val])), np.nanmedian(abs(R @ weights - ry[val]))])
        print(name, 'leave-lab-out median absolute strain/rate errors', np.round(folds, 6).tolist(), 'worst', np.round(np.max(folds, axis=0), 6).tolist())
        taus, weights = m._fit(design(eligible), cy[eligible], ry[eligible], dt, t)
        C, R = m._design(te, xe, *args)(taus)
        e = te['strain_0'][:, None] + C @ weights - g['te_clean']
        r = R @ weights - g['te_true_rate0']
        for label, mask in [('complete', ~missing), ('imputed', missing)]:
            if mask.any():
                print(name, label, 'test strain/rate RMSE', np.sqrt(np.mean(e[mask]**2)), np.sqrt(np.mean(r[mask]**2)))
        print(name, 'max absolute parameter errors (tau, hereditary, regime)', np.max(abs(taus - g['taus'])), np.max(abs(weights[:7] - g['theta'])), np.max(abs(weights[7:] - g['vp_theta'])))
    exact_args = (np.c_[g['stress_gains'], g['stress_biases']], *args[1:])
    C, R = m._design(te, xe, *exact_args)(g['taus'])
    w = np.r_[g['theta'], g['vp_theta']]
    if missing.any():
        e = te['strain_0'][:, None] + C @ w - g['te_clean']
        r = R @ w - g['te_true_rate0']
        print('Imputed test strain/rate RMSE with exact physical parameters and unchanged panel imputation', np.sqrt(np.mean(e[missing]**2)), np.sqrt(np.mean(r[missing]**2)))
