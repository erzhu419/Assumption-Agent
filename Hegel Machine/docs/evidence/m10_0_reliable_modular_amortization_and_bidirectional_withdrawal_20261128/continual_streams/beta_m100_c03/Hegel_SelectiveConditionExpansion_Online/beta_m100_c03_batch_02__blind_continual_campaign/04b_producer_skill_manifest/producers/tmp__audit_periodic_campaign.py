import pickle
import sys
import numpy as np

sys.path.insert(0, '/holdout_task')
from batch_solver import _PAY, solve_batch
from _verified_upstream import calibrate_upstream
from downstream_solver import _axis, _responses, solve_downstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    fixture = pickle.load(f)
for item in fixture:
    s = item['study']
    calibrated = [calibrate_upstream(s, _PAY), calibrate_upstream(s)]
    S, last = _responses(calibrated[0])
    pulses = item['campaign_diagnostics']
    W = np.column_stack([p['w_pulse'] for p in pulses])
    prediction = S @ W
    ns = len(s['mu_shifts'])
    for name, pay in [('staged', _PAY), ('sweep-only', None)]:
        Y = np.column_stack([calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']), pay)['y_obs_raw'] for p in pulses])
        e = (prediction-Y).reshape(ns, -1, 4, len(pulses))
        rms = lambda a, axis=None: np.sqrt(np.mean(a*a, axis=axis))
        print(name, 'worst shift/channel/pulse RMS', rms(e,1).max(), 'per-channel RMS', rms(e,(0,1,3)))
        print(name, 'worst whole pulse/shift RMS', rms(e,(1,2)).max(), 'peak abs', abs(e).max())
        Z = np.asarray(s['Z_nuisance'])
        beta = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T @ (prediction-Y))
        ep = (prediction-Y-Z@beta).reshape(e.shape)
        print(name, 'worst nuisance-profiled shift/channel/pulse RMS', rms(ep,1).max())
        assert rms(e,(1,2)).max() < 2e-5
    outs = [solve_batch(s), solve_downstream(calibrated[1])]
    print('calibration sensitivity: weights and terminal field max absolute change', *[np.max(abs(outs[0][key]-outs[1][key])) for key in ('weights', 'terminal_field')])
    for d in 'xy':
        k, v = (np.asarray(s[key+'_'+d]) for key in ('kappa','vel'))
        h = s['L'+d]/(len(k)-1)
        a = _axis(k,v,h).toarray()
        seam = (k[0]+k[-1])/(2*h**2)
        err = max(abs(a[0,-1]+seam+max(v[0],0)/h), abs(a[-1,0]+seam+max(-v[-1],0)/h), np.max(abs(a.sum(axis=1))))
        print(d, 'directional seam and constant-mode maximum error', err)
        assert err < 1e-12
