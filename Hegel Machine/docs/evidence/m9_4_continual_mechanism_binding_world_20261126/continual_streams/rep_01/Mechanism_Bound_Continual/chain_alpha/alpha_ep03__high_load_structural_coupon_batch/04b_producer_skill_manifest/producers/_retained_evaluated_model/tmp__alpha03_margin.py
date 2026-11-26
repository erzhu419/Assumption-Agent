import importlib.util
import pickle
import numpy as np
s = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep03_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    packets = pickle.load(f)
print('public datasets:', len(packets))
for packet in packets:
    ds = packet['dataset']
    cal = ds['calibration_load_cells']
    for c in np.unique(cal['lab_id']):
        mask = cal['lab_id'] == c
        u, v = cal['cmd_stress'][mask], cal['true_stress'][mask]
        coef = np.linalg.lstsq(np.c_[u, np.ones(len(u))], v, rcond=None)[0]
        residual = v - np.c_[u, np.ones(len(u))] @ coef
        print('calibration', c, 'gain/bias', np.round(coef, 6), 'max abs residual', np.max(abs(residual)))
    a, b, y, complete = m._prepare(ds)
    complete = complete[:len(y)]
    lab, stratum = (ds['train'][k] for k in ('lab_id', 'stratum'))
    e = np.empty_like(y[:, :-1]); rate = np.empty(len(y))
    for c in np.unique(lab):
        v = lab == c
        tau, theta = m._fit(a, y, complete & ~v)
        e[v] = (m._predict(a, tau, theta, y[:, -1], complete) - y[:, :-1])[v]
        rate[v] = (a.matrix(tau)[:, -1] @ theta - y[:, -1])[v]
    groups = [('all', np.ones(len(y), bool))]
    groups += [(f'lab{c}/stratum{k}', (lab == c) & (stratum == k)) for c in np.unique(lab) for k in np.unique(stratum)]
    groups += [(f'{name}/complete={flag}', mask & (complete == flag)) for name, mask in [('subyield', (a.work == 0) & (a.ov == 0)), ('hold+', a.ov > 0), ('hold-', a.ov < 0)] for flag in (False, True)]
    for name, g in groups:
        if g.any():
            z = e[g]
            print(name, g.sum(), 'bias/RMSE/MedAE/earlyRMSE/lateRMSE', np.round([z.mean(), np.sqrt(np.mean(z*z)), np.median(abs(z)), np.sqrt(np.mean(z[:,0]**2)), np.sqrt(np.mean(z[:,-1]**2))], 5))
    for j in np.argsort(np.mean(e*e, axis=1))[-8:][::-1]:
        print('large residual', j, 'lab/stratum', (lab[j], stratum[j]), 'complete', complete[j], 'rate residual', round(rate[j], 4), 'curve residual', np.round(e[j], 3))
    for name, panel in [('train', ds['train']), ('test', ds['test'])]:
        cmd = np.c_[panel['past_stress_cmd'], panel['stress_hold_cmd']]
        a_past, a_hold = m._physical_stresses(ds, panel)
        phys = np.c_[a_past, a_hold]
        margin = np.abs(np.abs(phys)-ds['yield_stress'])
        print(name, 'crossings changed by calibration', np.sum((np.abs(cmd)>ds['yield_stress']) != (np.abs(phys)>ds['yield_stress'])), 'nearest yield margin', margin.min())
