import importlib.util
import pickle
import numpy as np

p = '/holdout_task/alpha_m97_c03_batch_03_solver.py'
spec = importlib.util.spec_from_file_location('solver', p)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_03.pkl','rb'))[0]['dataset']
train, test = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
params = s._calibrate(d['calibration_load_cells'])
x, _ = s._align(train, test)
model = s._design(train, x, params, dt, t, True)
y = train['strain_curves_obs'] - train['strain_0'][:,None]
r = train['instant_rate_obs']
labs = train['lab_id']
missing = ~np.isfinite(train['coupon_features_raw']).all(axis=1)
hold = params[labs, 0]*train['stress_hold_cmd'] + params[labs, 1]
slices = {**{f'lab{i}': labs == i for i in range(4)}, **{f'stratum{i}':train['stratum'] == i for i in np.unique(train['stratum'])}, 'missing':missing, 'complete':~missing, 'negative':hold<0, 'positive':hold>=0, 'high_abs_stress':np.abs(hold)>d['crit_stress']}
rmse = lambda z: np.sqrt(np.mean(z*z))
for weight in (0.25, 0.5, 1.0, 2.0):
    ec, er, taus = np.empty_like(y), np.empty_like(r), []
    for lab in range(4):
        val = labs == lab
        tau, w = s._fit(model, y, r, dt, t, rate_weight=weight, row_mask=~val)
        c, rr = model(tau)
        ec[val], er[val] = (c @ w-y)[val], (rr @ w-r)[val]
        taus.append(tau)
    print('weight',weight,'taus',np.round(taus,5).tolist(), 'pooled', tuple(np.round([rmse(ec),rmse(er)],6)))
    print('slices', {k:tuple(np.round([rmse(ec[v]),rmse(er[v])],6)) for k,v in slices.items() if v.any()})
    print('per-horizon',np.round(np.sqrt(np.mean(ec**2,axis=0)),6).tolist())
cal = d['calibration_load_cells']
for lab in range(4):
    chosen = np.flatnonzero(cal['lab_id']==lab)
    for j in chosen:
        spike = {k:v.copy() for k,v in cal.items()}
        spike['true_stress'][j] += 3.2
        assert np.max(np.abs(s._calibrate(spike)-params)) < 0.00025, (lab,j)
    keep = (cal['lab_id']!=lab)
    keep[chosen[0]] = True
    sparse = {k:v[keep] for k,v in cal.items()}
    print('sparse',lab,'prefix delta',np.round(s._calibrate(sparse)[lab]-params[lab],7))
