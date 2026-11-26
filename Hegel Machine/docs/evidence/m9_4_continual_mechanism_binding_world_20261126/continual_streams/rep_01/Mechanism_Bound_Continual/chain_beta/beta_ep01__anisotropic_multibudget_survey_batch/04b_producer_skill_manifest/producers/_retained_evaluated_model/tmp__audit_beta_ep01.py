import importlib.util
import pathlib
import pickle
import numpy as np
from scipy.linalg import solve_sylvester

root = pathlib.Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'beta_ep01_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
a = pickle.loads((root / 'public_beta_ep01.pkl').read_bytes())[0]
s, gt = a['study'], a['gt']
ns = {}
exec((root / 'validate_public_schema.py').read_text().split('\n_d = ', 1)[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']

def axis(name):
    n = s['n'+name]
    h = s['L'+name]/n
    k, v, r = (s[key+'_'+name] for key in ('kappa', 'vel', 'reaction'))
    robin = s['robin_'+name]
    out = np.diag(r.copy())
    for i in range(n):
        for j in (i-1, i+1):
            if 0 <= j < n:
                flux = 2 / (1/k[i]+1/k[j]) / h**2
                out[i,i] += flux
                out[i,j] -= flux
        up = i-1 if v[i] >= 0 else i+1
        out[i,i] += abs(v[i])/h
        if 0 <= up < n:
            out[i,up] -= abs(v[i])/h
        if i == 0 or i == n-1:
            out[i,i] += 1 / (h * (1/robin + h/(2*k[i])))
    return out

A, B = axis('x'), axis('y')
out = m.solve_beta_ep01(s)
w = out['weights_by_budget'][1]
mu = s['mu_shifts'][-1]
field = solve_sylvester(A + mu*np.eye(s['nx']), B.T, np.einsum('m,mij->ij', w, s['source_modes']))
field_err = np.max(np.abs(field - out['terminal_field_mid_budget']))
print('Independent stencil/Sylvester terminal field max discrepancy:', field_err)
assert field_err < 1e-10

sweep = s['sensor_calibration_sweeps']
errors = {}
for bank in np.unique(sweep['bank_id']):
    idx = np.flatnonzero(sweep['bank_id'] == bank)
    ref, raw = sweep['ref_signal'], sweep['raw_readout']
    errs = []
    for i in idx:
        train = idx[idx != i]
        gain, offset = np.linalg.lstsq(np.column_stack((ref[train], np.ones(len(train)))), raw[train], rcond=None)[0]
        errs.append(abs(gain*ref[i] + offset - raw[i]))
    errors[int(bank)] = max(errs)
print('Leave-one-reference-out maximum raw readout error by bank:', errors)
assert max(errors.values()) < 1e-5

# Integer cap multiples allow a solution with no free variables; below/above test adjacent active faces.
cap = s['u_box']
t = dict(s, budget_grid=cap*np.array([2.-1e-6, 2., 2.+1e-6]))
stress = m.solve_beta_ep01(t)
checks = [verify(gt['S_tilde'], gt['r_y'], s['source_ridge'], cap, b, c) for b,c in zip(t['budget_grid'], stress['kkt_certificates'])]
print('Integer-budget neighborhood certificate gaps:', [c['gap'] for c in checks])
assert all(c['valid'] for c in checks)
