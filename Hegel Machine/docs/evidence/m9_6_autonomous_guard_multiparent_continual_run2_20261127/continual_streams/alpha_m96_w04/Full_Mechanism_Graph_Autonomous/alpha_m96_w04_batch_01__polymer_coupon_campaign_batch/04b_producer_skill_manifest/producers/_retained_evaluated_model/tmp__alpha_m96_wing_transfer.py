import importlib.util
import pickle

import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m96_w04_batch_01_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
d = pickle.load(open(root + 'public_alpha_m96_w04_batch_01.pkl', 'rb'))[0]['dataset']
tr, te = d['train'], d['test']
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
wing = lab // 2
cal = solver._calibration(d['calibration_load_cells'])
x = solver._aligned(tr, te)[:len(lab)]
y = np.c_[tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']]
raw = tr['coupon_features_raw']

def model(mask, features):
    return solver._Creep({k: v[mask] for k,v in tr.items()}, features[mask], cal, d['past_dt'], d['eval_times'])

for group_name, group in [('wing', wing), ('lab', lab)]:
    for label in np.unique(group):
        fit = group != label
        test = ~fit
        raw_fill = np.where(np.isfinite(raw), raw, np.nanmedian(raw[fit], axis=0))
        for name, select, features, use_rate in [('baseline', fit, x, True), ('unaligned', fit, raw_fill, True), ('complete_fit', fit & complete, x, True), ('curve_only', fit, x, False)]:
            target = y[select].copy()
            if not use_rate:
                target[:, -1] = np.nan
            tau, theta = solver._fit(model(select, features), target)
            err = model(test, features).basis(tau)[0] @ theta - y[test]
            core = np.abs(err[complete[test]])
            missing = np.abs(err[~complete[test], :-1])
            cells = [np.nanmedian(abs(err[(lab[test] == l) & (tr['stratum'][test] == s), :-1])) for l,s in np.unique(np.c_[lab[test], tr['stratum'][test]], axis=0)]
            print(group_name, label, name, 'tau', np.round(tau, 5), 'complete curve/final/rate MedAE', np.round([np.nanmedian(core[:, :-1]), np.nanmedian(core[:, -2]), np.nanmedian(core[:, -1])], 6), 'missing curve MedAE', round(np.nanmedian(missing), 6) if missing.size else None, 'worst cell MedAE', round(max(cells), 6))
