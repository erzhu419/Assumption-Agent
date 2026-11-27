import itertools, json, pickle, sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _axis, _responses
from batch_solver import solve_batch
root = Path('/holdout_task')
with (root / 'public_batch.pkl').open('rb') as f:
    item = pickle.load(f)[0]
s = item['study']
nx, ny = s['nx'], s['ny']
ij = s['sensor_ij']
shifts = s['mu_shifts']
W = np.column_stack([p['w_pulse'] for p in item['campaign_diagnostics']])
Y = np.column_stack([calibrate_upstream(dict(s, y_obs_raw=p['y_pulse_raw']))['y_obs_raw'] for p in item['campaign_diagnostics']])
Y = Y.reshape(len(shifts), len(ij), -1)
S, _ = _responses(s)
production = (S @ W).reshape(Y.shape)

def axis(a, mode):
    k, v = s['kappa_'+a], s['vel_'+a]
    n = len(k)
    h = s['L'+a]/(n-1)
    result = np.zeros((n,n))
    for i in range(n):
        if i in (0,n-1) and mode != 'periodic':
            inward = 1 if i == 0 else n-2
            q = k[i]/h**2
            result[i,inward] = -q
            result[i,i] = q + (s['robin_'+a]/h if mode == 'robin' else 0)
            if mode == 'diffusion_seam':
                q = (k[0]+k[-1])/(2*h**2)
                result[i,n-1-i] = -q
                result[i,i] += q
            continue
        for j, adv in [((i-1)%n,max(v[i],0)), ((i+1)%n,max(-v[i],0))]:
            q = (k[i]+k[j])/(2*h**2)+adv/h
            result[i,j] -= q
            result[i,i] += q
    return result

conservation = 0.0
for a in 'xy':
    k, v = s['kappa_'+a], s['vel_'+a]
    operator = _axis(k, v, s['L'+a]/(len(k)-1))
    conservation = max(conservation, float(np.max(np.abs(operator @ np.ones(len(k))))))
assert conservation < 1e-10
rhs = np.stack([m.ravel(order='F') for m in s['source_modes']], axis=1) @ W
I = np.eye(nx*ny)
reaction = np.diag(s['V_reaction_2d'].ravel(order='F'))
idx = ij[:,0]+nx*ij[:,1]
modes = ('periodic','robin','reflect','diffusion_seam')
axis_x = {m:axis('x',m) for m in modes}
axis_y = {m:axis('y',m) for m in modes}
errors = {}
for x,y in itertools.product(modes,repeat=2):
    L = np.kron(np.eye(ny),axis_x[x])+np.kron(axis_y[y],np.eye(nx))+reaction
    pred = np.stack([solve(L+mu*I,rhs)[idx] for mu in shifts])
    if (x,y)==('periodic','periodic'):
        np.testing.assert_allclose(pred,production,rtol=0,atol=1e-12)
    errors[x,y] = pred-Y
rmse = lambda v: float(np.sqrt(np.mean(v**2)))
seam = np.any((ij==0)|(ij==[nx-1,ny-1]),axis=1)
worst = 0.0
min_ratio = float('inf')
seam_worst = 0.0
unique = np.unique(ij,axis=0)
for sensor in unique:
    held = np.all(ij==sensor,axis=1)
    ranked = sorted(errors,key=lambda m: rmse(errors[m][:,~held,:]))
    assert ranked[0]==('periodic','periodic'), (sensor,ranked[:2])
    a,b = [rmse(errors[m][:,held,:]) for m in ranked[:2]]
    worst = max(worst,a)
    min_ratio = min(min_ratio,b/max(a,1e-15))
    if np.any(seam&held):
        seam_worst = max(seam_worst,a)
assert worst < 1e-5 and seam_worst > 0 and min_ratio > 5
out = solve_batch(s)
assert np.isfinite(out['terminal_field']).all()
print(json.dumps({'nogood_id':'ng_t04','audit':'leave-one-sensor-location-out across all shifts and tracer pulses','sensor_locations':len(unique),'selected_boundary':['periodic','periodic'],'worst_held_sensor_rmse':worst,'worst_held_seam_sensor_rmse':seam_worst,'minimum_runner_up_rmse_ratio':min_ratio,'max_constant_state_transport_residual':conservation}))
