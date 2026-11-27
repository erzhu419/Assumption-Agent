import json, pickle, sys
import numpy as np
from scipy.sparse import coo_matrix
sys.path.insert(0, '/holdout_task')
import downstream_solver as dn
from _verified_upstream import calibrate_upstream

item = pickle.load(open('public_batch.pkl', 'rb'))[0]
s = item['study']
pay = json.load(open('_upstream_param_payload.json'))
pulses = item['campaign_diagnostics']
W = np.column_stack([p['w_pulse'] for p in pulses])
Y = np.column_stack([dn._calibrate(calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']), pay)) for p in pulses])
Z = s['Z_nuisance']
Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
shape = (len(s['mu_shifts']), len(s['sensor_ij']), len(pulses))
axis = dn._axis

def ablation(name):
    def assemble(kappa, vel, h):
        a = axis(kappa, vel, h).tolil()
        for i, j, sign in ((0, len(kappa)-1, 1), (len(kappa)-1, 0, -1)):
            original = -a[i,j]
            rate = {'remove_wrap': 0., 'no_seam_advection': (kappa[i]+kappa[j])/(2*h*h),
                    'local_seam_diffusivity': kappa[i]/(h*h)+max(sign*vel[i],0.)/h}[name]
            a[i,j] = -rate
            a[i,i] += rate-original
        return a.tocsc()
    return assemble

banks = np.asarray(s['row_bank_id']).reshape(shape[:2])
results = {}
for name in ('exact_periodic', 'remove_wrap', 'no_seam_advection', 'local_seam_diffusivity'):
    dn._axis = axis if name == 'exact_periodic' else ablation(name)
    S, _ = dn._forward(s)
    e = S @ W - Y
    projected = e - Z @ (Pz @ e)
    raw = e.reshape(shape)
    worst = max(float(np.sqrt(np.mean(raw[k, banks[k]==b, p]**2))) for k in range(shape[0]) for b in np.unique(banks[k]) for p in range(shape[2]))
    by_pulse = np.sqrt(np.mean(e**2, axis=0))
    results[name] = worst
    print(json.dumps({'model': name, 'pulse_rmse': by_pulse.tolist(), 'projected_pulse_rmse': np.sqrt(np.mean(projected**2,axis=0)).tolist(), 'worst_pulse_shift_bank_rmse': worst}))
dn._axis = axis
assert results['exact_periodic'] < 1e-5, results
assert all(results[a] > 10*results['exact_periodic'] for a in results if a != 'exact_periodic'), results
print('ng_t06: measured seam ablations verify the opposite-wall term, shared seam diffusion, and endpoint upwinding on the exact staged candidate')
