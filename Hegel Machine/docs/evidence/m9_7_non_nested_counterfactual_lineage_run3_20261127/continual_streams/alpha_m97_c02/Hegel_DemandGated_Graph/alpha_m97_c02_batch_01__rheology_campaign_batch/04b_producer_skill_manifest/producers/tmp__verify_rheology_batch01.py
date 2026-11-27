import importlib.util
import pickle
import numpy as np

path = '/holdout_task/alpha_m97_c02_batch_01_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c02_batch_01.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, gt = item['dataset'], item['gt']
out = m.solve_alpha_m97_c02_batch_01(d)
rmse = lambda e: np.sqrt(np.mean(e**2))
cal = m._calibrate(d['calibration_load_cells'])
common = (d['past_dt'], d['eval_times'], d['crit_stress'], d['shear_banding_index'])
for key, ref in [('estimated_taus','taus'), ('hereditary_weights','theta'), ('regime_mode_weights','vp_theta'), ('predicted_strain','te_clean'), ('initial_rate_pred','te_true_rate0')]:
    err = out[key] - gt[ref]
    print(key, 'rmse', rmse(err), 'max_abs', np.max(np.abs(err)))
    if err.size < 10:
        print('  signed errors', err)
test = d['test']
past, hold = m._physical(test, cal)
print('calibrated stress max abs', np.max(np.abs(past-gt['te_hist_true'])), np.max(np.abs(hold-gt['te_hold_true'])))
z = np.zeros((len(hold), 2))
tau = out['estimated_taus']
for step, dt in enumerate(d['past_dt']):
    decay = np.exp(-dt/tau)
    z = decay*z + (1-decay)*past[:, step, None]
active = ((np.maximum(abs(past)-d['crit_stress'],0)**2)@d['past_dt']) > 0
for lab in range(4):
    for mode in (False, True):
        take = (test['lab_id'] == lab) & (active == mode)
        if take.any():
            print('public lab/active', lab, int(mode), 'n', take.sum(), 'curve/rate RMSE', rmse((out['predicted_strain']-gt['te_clean'])[take]), rmse((out['initial_rate_pred']-gt['te_true_rate0'])[take]))
x = m._align(d['train'], test)[-len(hold):]
a,b = m._Design(past, hold, x, *common).matrices(tau)
assert np.allclose(b[:, :2], (hold[:,None]-z)/tau, atol=1e-12, rtol=0)
for shear in (.50, .50001):
    curve0, rate = m._Design(past, hold, x, common[0], np.array([0., 1.e-7]), common[2], shear).matrices(tau)
    assert np.all(curve0[:, 0] == 0.)
    assert np.allclose(curve0[:, 1]/1.e-7, rate, atol=8e-6, rtol=1e-6)
    if shear > .50:
        assert np.all(curve0[:, :, 7:] == 0) and np.all(rate[:, 7:] == 0)
train = d['train']
for lab in range(4):
    keep = train['lab_id'] != lab
    ns = {k:v[keep] for k,v in train.items()}
    cv = dict(d, train=ns, test={k:np.concatenate((v[~keep],test[k])) for k,v in train.items() if k in test})
    pred = m.solve_alpha_m97_c02_batch_01(cv)
    n = int((~keep).sum())
    ec = pred['predicted_strain'][:n]-train['strain_curves_obs'][~keep]
    er = pred['initial_rate_pred'][:n]-train['instant_rate_obs'][~keep]
    print('held-out lab', lab, 'curve/rate RMSE', rmse(ec), rmse(er), 'curve/rate median abs', np.median(abs(ec)), np.median(abs(er)), 'worst horizon RMSE', np.max(np.sqrt(np.mean(ec**2, axis=0))))
rows = d['calibration_load_cells']
spiked = {k:v.copy() for k,v in rows.items()}
for lab in range(4):
    idx = np.flatnonzero(rows['lab_id'] == lab)
    spiked['true_stress'][idx[np.argmin(rows['cmd_stress'][idx])]] += 5.
spike_err = np.max(abs(m._calibrate(spiked)-cal))
print('single EMI per channel: max coefficient shift', spike_err)
assert spike_err < .001
for num in (0, 1):
    keep = rows['lab_id'] != 0
    keep[np.flatnonzero(~keep)[:num]] = True
    fit = m._calibrate({k:v[keep] for k,v in rows.items()})
    assert np.all(np.isfinite(fit)) and np.allclose(fit[1:], cal[1:])
    print('history fallback, lab 0 in-batch rows', num, 'coefficients', fit[0])
