import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch07', root + 'alpha_w03_batch_07_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root + 'public_alpha_w03_batch_07.pkl', 'rb'))[0]['dataset']
tr, te = d['train'], d['test']
labs = tr['lab_id']
base = m._calibrate
fits = base(d['calibration_load_cells'], labs)
for lab, f in fits.items():
    x, y = m._checks(d['calibration_load_cells'], lab)
    print('lab', lab, 'absolute spot residuals', np.round(np.sort(abs(y-np.polyval(f,x))), 6))

# Fixed calibration challengers; no held-out strain or rate used to choose a fit.
def calibration(mode):
    def fit(table, labels):
        result = base(table, labels)
        for lab in (2, 3):
            if lab not in result:
                continue
            if mode == 'identity Wing B':
                result[lab] = np.array([1., 0.])
            elif mode == 'one-spot bias adjustment':
                x, y = m._checks(table, lab)
                r = y - np.polyval(result[lab], x)
                result[lab] = result[lab] + [0., r[np.argmin(abs(r))]]
        return result
    return fit

_, h, p = m._prep(d)
h = h[:len(labs)]
p = p[:len(labs)]
miss = np.isnan(tr['coupon_features_raw']).any(axis=1)
slices = {f'lab {j}': labs == j for j in np.unique(labs)}
slices.update({'missing': miss, 'complete': ~miss, 'high |hold|': abs(h) >= np.quantile(abs(h), .85), 'stress reversal': np.sign(h) != np.sign(p[:, -1])})
for mode in ('archive', 'one-spot bias adjustment', 'identity Wing B'):
    m._calibrate = calibration(mode)
    ec = np.empty_like(tr['strain_curves_obs']); er = np.empty_like(tr['instant_rate_obs'])
    for lab in np.unique(labs):
        val = labs == lab
        # Keep the reference panel fixed. Only unobserved covariates enter test.
        fold = dict(d, train={k:v[~val] for k,v in tr.items()}, test={k:np.concatenate((tr[k][val],v)) for k,v in te.items() if k in tr and k not in ('strain_curves_obs',)})
        out = m.solve_alpha_w03_batch_07(fold)
        ec[val] = out['predicted_strain'][:val.sum()] - tr['strain_curves_obs'][val]
        # Score the constitutive rate before any update from the observed test rate.
        xf, hf, pf = m._prep(fold)
        start = int((~val).sum())
        held = slice(start, start + int(val.sum()))
        _, r = m._design(out['estimated_taus'], xf[held], hf[held], pf[held],
                         d['past_dt'], d['eval_times'], d['yield_stress'])
        w = np.r_[out['hereditary_weights'], out['viscoplastic_weights']]
        er[val] = r @ w - tr['instant_rate_obs'][val]
    print(mode)
    for name, mask in slices.items():
        if mask.any():
            print(name, int(mask.sum()), 'p90 curve/prior rate', np.round([np.quantile(abs(ec[mask]), .9), np.quantile(abs(er[mask]), .9)], 6), 'signed median curve/rate', np.round([np.median(ec[mask]), np.median(er[mask])], 6))
    if mode == 'archive':
        for lab in np.unique(labs):
            mask = labs == lab
            corr = np.corrcoef(np.column_stack((ec[mask,-1], er[mask], h[mask], h[mask]-p[mask,-1], h[mask]**2)), rowvar=False)
            print('lab', lab, 'late curve/rate residual correlation with hold, last stress gap, squared hold', np.round(corr[:2,2:], 3))
