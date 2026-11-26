import importlib.util
import pickle
import numpy as np

p = '/holdout_task/alpha_w01_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('solver', p)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_w01_batch_04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
dt, time, sy = d['past_dt'], d['eval_times'], d['yield_stress']
maps = s._calibrate(d['calibration_load_cells'])

def sub(block, mask):
    return {k: np.asarray(v)[mask] for k, v in block.items()}

def rmse(x):
    return float(np.sqrt(np.mean(x*x)))

worst = {}
for wing in (0, 1):
    loc = tr['lab_id']//2 == wing
    fit, val = sub(tr, ~loc), sub(tr, loc)
    panel = {k: np.concatenate((val[k], te[k])) for k in te if k in val}
    x, complete, n = s._alignment(fit, panel)
    m = len(val['lab_id'])
    keep, good = complete[:n], complete[n:n+m]
    fit = sub(fit, keep)
    xi = s._inputs(fit, x[:n][keep], maps, dt, sy)
    vi = s._inputs(val, x[n:n+m], maps, dt, sy)
    gate = (vi[-2] != 0).astype(int) + 2*(vi[-1] != 0)
    for plastic in (False, True):
        tau, w = s._fit(fit, xi, dt, time, plastic=plastic)
        a, b = s._design(tau, vi, dt, time)
        r = b @ w - val['instant_rate_obs']
        e = val['strain_0'][:, None] + a @ w - val['strain_curves_obs']
        ec = e - np.where(~good & np.isfinite(r), r, 0.)[:, None]*time
        print('wing', wing, 'plastic', plastic, 'taus', tau.round(4), 'plastic weights', w[7:].round(5), 'rate RMSE (without using held-out rate)', round(rmse(r), 5))
        for j in range(4):
            mask = gate == j
            if mask.any():
                metric = rmse(ec[mask])
                worst[plastic] = max(worst.get(plastic, 0.), metric)
                print('  regime', j, 'n', mask.sum(), 'curve RMSE', round(metric, 5), 'curve MAE', round(np.mean(abs(ec[mask])), 5), 'rate median abs', round(np.median(abs(r[mask])), 5))
        if plastic and (~good).any():
            print('  missing-feature curve RMSE without/with rate correction', round(rmse(e[~good]), 5), round(rmse(ec[~good]), 5))
print('maximum leave-one-wing-out regime RMSE: linear', worst[False], 'gated', worst[True])
