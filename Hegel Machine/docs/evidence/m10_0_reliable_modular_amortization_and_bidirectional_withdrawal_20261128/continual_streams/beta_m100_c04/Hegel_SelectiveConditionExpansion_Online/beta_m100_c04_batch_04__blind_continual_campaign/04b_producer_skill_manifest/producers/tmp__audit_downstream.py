import pickle
import runpy
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path[:0] = ['/holdout_task', '/tmp']
from _verified_upstream import calibrate_upstream
from downstream_solver import _forward, solve_downstream
from batch_solver import solve_batch
from probe_boundary import forward as trial_forward

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
pulses = item['campaign_diagnostics']
weights = np.stack([p['w_pulse'] for p in pulses])
modes = np.einsum('ab,bij->aij', weights, s['source_modes'])
y = np.column_stack([calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']))['y_obs_raw'] for p in pulses])
S, _ = _forward(s)
fixed = S @ weights.T - y
k = s['kappa_x']; v = s['vel_x']; h = s['Lx'] / (s['nx'] - 1)
seam = (k[0] + k[-1]) / 2
p0 = np.array([((k[0]+k[1])/2+h*max(-v[0],0))/k[0], 0, (seam+h*max(v[0],0))/k[0], ((k[-1]+k[-2])/2+h*max(v[-1],0))/k[-1], 0, (seam+h*max(-v[-1],0))/k[-1], 1, 0, 0, 1, 0, 0])
Z = s['Z_nuisance']
ns = len(s['sensor_ij']); nshifts = len(s['mu_shifts']); shift_idx = np.repeat(np.arange(nshifts), ns)
xy = np.asarray(s['sensor_ij']); xwall = np.tile(np.isin(xy[:,0], [0,s['nx']-1]), nshifts); ywall = np.tile(np.isin(xy[:,1], [0,s['ny']-1]), nshifts)
def rms(e):
    return np.sqrt(np.mean(e**2, axis=0))
def profile(z, e):
    return np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ e)

def error(p):
    return trial_forward(s, p, modes)[0] - y

np.testing.assert_allclose(error(p0), fixed, atol=1e-13, rtol=0)
worst = np.zeros((3, 2))
low = np.tile([.001, 0., 0.], 4); high = np.tile([6., 20., 5.], 4)
for hold in range(nshifts):
    train = shift_idx != hold; test = ~train
    ztr, zte = Z[train], Z[test]
    for j, cols in enumerate(([], [0,2,3,5], list(range(12)))):
        p = p0.copy()
        if cols:
            def loss(x):
                trial = p0.copy(); trial[cols] = x
                d = error(trial)[train]
                return (d - ztr @ profile(ztr, d)).ravel()
            fit = least_squares(loss, p0[cols], bounds=(low[cols], high[cols]), gtol=1e-13, ftol=1e-12, xtol=1e-12, max_nfev=120)
            p[cols] = fit.x
        d = error(p) if j else fixed
        raw, adjusted = rms(d[test]), rms(d[test] - zte @ profile(ztr, d[train]))
        worst[j] = np.maximum(worst[j], [max(raw), max(adjusted)])
        print(f'shift {hold} model {j}: max pulse RMSE raw {max(raw):.3e}, train-only nuisance adjusted {max(adjusted):.3e}')
print('Worst shift/pulse by model [fixed, fitted seam, full wall], columns [raw, adjusted]:', worst.tolist())
for name, gate in [('x-wall', xwall), ('y-wall', ywall), ('interior', ~(xwall | ywall))]:
    if gate.any():
        print(f'{name} worst pulse RMSE {max(rms(fixed[gate])):.3e}')
assert worst[0, 0] < 4e-6
out = solve_batch(s); direct = solve_downstream(calibrate_upstream(s))
for field in ('weights', 'terminal_field'):
    np.testing.assert_array_equal(out[field], direct[field])
assert out['kkt_certificate'] == direct['kkt_certificate']
sys.argv = ['/holdout_task/validate_public_schema.py', '/holdout_task/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
