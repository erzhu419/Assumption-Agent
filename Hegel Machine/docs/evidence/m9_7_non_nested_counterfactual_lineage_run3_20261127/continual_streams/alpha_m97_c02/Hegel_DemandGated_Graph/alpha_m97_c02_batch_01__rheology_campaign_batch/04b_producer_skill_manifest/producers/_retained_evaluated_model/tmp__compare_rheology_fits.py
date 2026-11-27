import importlib.util
import pickle
import numpy as np

p = '/holdout_task/alpha_m97_c02_batch_01_solver.py'
s = importlib.util.spec_from_file_location('model', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, gt = item['dataset'], item['gt']
tr, te = d['train'], d['test']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
cal = m._calibrate(d['calibration_load_cells'])
x = m._align(tr, te)
common = tuple(d[k] for k in ('past_dt', 'eval_times', 'crit_stress', 'shear_banding_index'))
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
test_complete = np.isfinite(te['coupon_features_raw']).all(axis=1)

def design(split, features, mask):
    return m._Design(*(v[mask] for v in m._physical(split, cal)), features[mask], *common)

n = len(y)
train_design = design(tr, x[:n], np.ones(n, bool))
test_design = design(te, x[n:], np.ones(len(te['lab_id']), bool))
rmse = lambda z: np.sqrt(np.mean(np.square(z)))

for name in ('baseline','complete','complete_refit'):
    errors = []
    for lab in (-1, 0, 1, 2, 3):
        take = (tr['lab_id'] != lab) if lab >= 0 else np.ones(n, bool)
        if name != 'baseline':
            take &= complete
        model = design(tr, x[:n], take)
        yy, rr = y[take], r[take]
        tau, w = m._fit(model, yy, rr)
        if name == 'complete_refit':
            a, b = model.matrices(tau)
            def retain(obs, fit):
                e = obs-fit
                center = np.nanmedian(e)
                sigma = 1.4826*np.nanmedian(abs(e-center))
                return np.where(abs(e-center) <= 4*max(sigma, 1e-5), obs, np.nan)
            a_target, b_target = retain(yy, a@w), retain(rr, b@w)
            tau, w = m._fit(model, a_target, b_target)
        if lab < 0:
            a,b = test_design.matrices(tau)
            ec = te['strain_0'][:,None] + a@w - gt['te_clean']
            er = b@w - gt['te_true_rate0']
            print(name, 'public tau max rel err', np.max(abs(tau/gt['taus']-1)), 'hw/rw max abs', np.max(abs(w[:7]-gt['theta'])), np.max(abs(w[7:]-gt['vp_theta'])), 'strain/rate RMSE', rmse(ec),rmse(er), 'complete', rmse(ec[test_complete]),rmse(er[test_complete]))
        else:
            q = (tr['lab_id'] == lab) & complete
            a,b = train_design.matrices(tau)
            ec, er = (a@w-y)[q], (b@w-r)[q]
            errors.append([np.median(abs(ec)),np.median(abs(er)),rmse(np.clip(ec,-.06,.06)),rmse(er)])
    print(name, 'worst held-out lab complete rows: curve/rate median abs, capped curve RMSE, rate RMSE', np.max(errors,axis=0))
