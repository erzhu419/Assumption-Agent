import json
import pickle
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _box_simplex, _responses
from _verified_upstream import calibrate_upstream

root = Path('/holdout_task')
item = pickle.loads((root/'public_batch.pkl').read_bytes())[0]
st = item['study']
s = calibrate_upstream(st)
out = solve_batch(st)
w = out['weights']
cert = out['kkt_certificate']
nx, ny = s['nx'], s['ny']
# Independent dense finite-volume reference, with explicit torus indexing.
L = np.diag(s['V_reaction_2d'].T.reshape(-1))
for j in range(ny):
    for i in range(nx):
        p = i + nx*j
        for d, a, n in [('x', i, nx), ('y', j, ny)]:
            h = s['L'+d]/(n-1)
            kap, vel = s['kappa_'+d], s['vel_'+d]
            for sign in [-1, 1]:
                b = (a+sign) % n
                q = b+nx*j if d == 'x' else i+nx*b
                rate = (kap[a]+kap[b])/(2*h*h) + max(0., -sign*vel[a])/h
                L[p,p] += rate
                L[p,q] -= rate
rhs = np.column_stack([mode.ravel(order='F') for mode in s['source_modes']])
obs = np.array([i+nx*j for i,j in s['sensor_ij']])
fields = [np.linalg.solve(L+mu*np.eye(nx*ny),rhs) for mu in s['mu_shifts']]
design = np.vstack([f[obs] for f in fields])
actual, _ = _responses(s)
assert np.allclose(actual,design,rtol=1e-11,atol=1e-12)
terminal = out['terminal_field'].ravel(order='F')
assert np.allclose((L+s['mu_shifts'][-1]*np.eye(nx*ny))@terminal,rhs@w,atol=1e-10)
Z = s['Z_nuisance']; y = s['y_obs_raw']; cap = s['u_box']; budget = s['total_budget']; ridge = s['source_ridge']
P = np.eye(len(y))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T)
D = P@design; r = D@w-P@y
g = D.T@r + ridge*w
lo, hi, nu = np.array(cert['lam_lower']), np.array(cert['lam_upper']), cert['nu_sum']
J = .5*(r@r+ridge*(w@w)); lb = J-g@w+nu*budget+cap*np.minimum(0,g-nu).sum()
checks = dict(forward=np.max(np.abs(actual-design)), field=np.max(np.abs(terminal-fields[-1]@w)), sum=abs(w.sum()-budget), stationarity=np.max(np.abs(g-nu-lo+hi)), complementarity=max(np.max(np.abs(w*lo)),np.max(np.abs((cap-w)*hi))), objective=abs(J-cert['primal_objective']), dual=abs(lb-cert['dual_lower_bound']), gap=abs(J-lb))
assert min(w) >= 0 and max(w) <= cap and min(lo) >= 0 and min(hi) >= 0
assert np.array_equal(w,cert['weights']) and max(checks.values()) < 1e-10
print('public independent errors:', checks)

# Re-encode the very same calibrated data with a different rosette sweep.
sw = st['sensor_calibration_sweeps']
newgain, newoffset = np.array([.92,1.1,.73,1.32]), np.array([.04,-.03,.12,.1])
newst = {**st, 'sensor_calibration_sweeps':{**sw, 'raw_rosette':sw['ref_rosette']*newgain+newoffset}, 'y_obs_raw':(y.reshape(-1,4)*newgain+newoffset).ravel()}
assert np.allclose(solve_batch(newst)['weights'],w,rtol=0,atol=1e-11)

rng = np.random.default_rng(9)
worst = 0.
for n in (1,2,8,32):
    for trial in range(24):
        A = rng.normal(size=(n,n)); H = A.T@A+.04*np.eye(n)
        c = rng.normal(size=n)*(20 if trial%2 else 1); u = .42
        for b in (0.,u*n,u*min(2,n),u*n/2,rng.uniform(0,u*n)):
            x,t = _box_simplex(H,c,u,b)
            d = H@x+c-t
            error = max(abs(x.sum()-b),max(0,-min(x),max(x)-u),abs(d@x-u*np.minimum(0,d).sum()))
            worst = max(worst,error)
assert worst < 1e-9
print('480 active-set stress cases; worst feasibility/tangent gap:',worst)

rows = json.loads(Path('/tmp/hegel_continual_archive/archive_apparatus_log.json').read_text())
for row in [rows[2],rows[-2],rows[-1]]:
    print('telemetry metadata:',{k:v for k,v in row.items() if k!='facility_telemetry_trace'})
