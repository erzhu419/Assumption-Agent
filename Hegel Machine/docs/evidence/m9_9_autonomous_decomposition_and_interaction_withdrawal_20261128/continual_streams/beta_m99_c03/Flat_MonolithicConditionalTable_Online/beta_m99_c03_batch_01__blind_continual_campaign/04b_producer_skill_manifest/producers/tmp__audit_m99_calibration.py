import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

p = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(p)
p.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)

def fit(kind, x, y):
    if kind == 'bank-diagonal':
        coef = np.array([np.linalg.lstsq(np.c_[np.ones(len(x)), x[:, j]], y[:, j], rcond=None)[0] for j in range(4)])
        return lambda v: coef[:, 0] + v * coef[:, 1]
    if kind == 'bank-rational':
        params = []
        for j in range(4):
            v, z = x[:, j], y[:, j]
            b, a = np.linalg.lstsq(np.c_[np.ones(len(v)), v], z, rcond=None)[0]
            q = least_squares(lambda q: q[0] + q[1]*v/(1+q[2]*v*v)-z, [b,a,0.], bounds=([-np.inf,-np.inf,0.],[np.inf,np.inf,np.inf]), gtol=1e-13).x
            params.append(q)
        b, a, d = np.asarray(params).T
        return lambda v: b + a*v/(1+d*v*v)
    def features(v):
        terms = [np.ones(len(v)), *v.T]
        if kind == 'interactions':
            terms += [v[:,i]*v[:,j] for i in range(4) for j in range(i,4)]
            terms += [v[:,i]*v[:,j]*v[:,k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4)]
        return np.array(terms).T
    coef = np.linalg.lstsq(features(x), y, rcond=None)[0]
    return lambda v: features(v) @ coef

for item in items:
    s = item['study']; cal = s['sensor_calibration_sweeps']
    x, y = cal['ref_rosette'], cal['raw_rosette']
    S, _ = m._forward(s)
    expected = np.stack([(S @ d['w_pulse']).reshape(-1,4) for d in item['campaign_diagnostics']])
    observed = np.stack([d['y_pulse_raw'].reshape(-1,4) for d in item['campaign_diagnostics']])
    folds = np.array_split(np.arange(len(x)), 8)
    for kind in ('bank-diagonal', 'affine', 'interactions', 'bank-rational'):
        err = np.empty_like(y)
        for hold in folds:
            keep = np.setdiff1d(np.arange(len(x)), hold)
            err[hold] = fit(kind, x[keep], y[keep])(x[hold])-y[hold]
        pred = fit(kind, x, y)(expected.reshape(-1,4)).reshape(expected.shape)
        rms = lambda e, axis: np.sqrt(np.mean(e**2,axis=axis))
        p = (pred-observed).reshape(len(expected),len(s['mu_shifts']),-1,4)
        print(kind, 'sweep worst fold/channel', format(max(rms(err[h],0).max() for h in folds),'.4g'), 'pulse worst pulse/shift/bank', format(rms(p,2).max(),'.4g'), 'pulse bank RMS', rms(p,(0,1,2)))
    banks = cal['single_bank_sweeps']; z = s['Z_nuisance']
    op = np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    ep = np.stack([m._calibrated_observations({**s,'y_obs_raw':d['y_pulse_raw']})-S@d['w_pulse'] for d in item['campaign_diagnostics']])
    print('full affine calibrated pulse worst bank RMS before/after nuisance', *(np.sqrt(np.mean(e.reshape(-1,4)**2,axis=0)).max() for e in (ep,ep-(z@(op@ep.T)).T)))
    print('response/nuisance spectral fraction', np.linalg.norm(z@(op@S),2)/np.linalg.norm(S,2), 'projected design relative singular spectrum', (lambda t:t/t[0])(np.linalg.svd(S-z@(op@S),compute_uv=False)))
    for j in range(4):
        i = banks['bank_id']==j
        a = np.polyfit(banks['ref_signal'][i], banks['raw_readout'][i], 1)
        print('bank', j, 'independent sweep slope/intercept minus rosette', a-np.polyfit(x[:,j],y[:,j],1))
