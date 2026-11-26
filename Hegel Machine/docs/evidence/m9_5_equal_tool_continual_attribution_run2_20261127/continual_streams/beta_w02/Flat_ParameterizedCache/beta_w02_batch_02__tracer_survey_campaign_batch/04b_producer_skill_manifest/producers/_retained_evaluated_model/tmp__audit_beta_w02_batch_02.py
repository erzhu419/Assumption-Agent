import importlib.util
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root / 'beta_w02_batch_02_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
case = pickle.loads((root / 'public_beta_w02_batch_02.pkl').read_bytes())[0]
s, ref = case['study'], case['gt']

def axis_diffusion(axis):
    n = s['n' + axis]
    h = s['L' + axis] / n
    k, r = s['kappa_' + axis], s['robin_' + axis]
    face = 2 / (1/k[:-1] + 1/k[1:]) / h**2
    leak = r / (h * (1 + 0.5*h*r/k[[0, -1]]))
    d = np.r_[face, 0] + np.r_[0, face]
    d[[0, -1]] += leak
    return diags([-face, d, -face], [-1, 0, 1], format='csc')

axes = []
for a in ('x', 'y'):
    n = s['n' + a]
    L, k, v, r = (s[key] for key in ('L' + a, 'kappa_' + a, 'vel_' + a, 'robin_' + a))
    diff = axis_diffusion(a)
    d0 = m._build_1d(L, n, k, 0*v, r, np.zeros(n))
    names = sorted(range(3), key=lambda i: len(d0[i]))
    mid = names[-1]
    np.testing.assert_allclose(d0[mid], diff.diagonal(), atol=2e-12, rtol=0)
    for i in names[:-1]:
        np.testing.assert_allclose(d0[i], diff.diagonal(1), atol=2e-12, rtol=0)
    vel = v / (L/n)
    adv = diags([-np.maximum(vel[1:],0), abs(vel), np.minimum(vel[:-1],0)], [-1,0,1], format='csc')
    # The base axis reactions are additive to the separate 2D reaction field.
    axes.append(diff + adv + diags(s['reaction_' + a], format='csc'))
nx, ny = s['nx'], s['ny']
A = (kron(axes[0], eye(ny)) + kron(eye(nx), axes[1]) + diags(s['V_reaction_2d'].ravel())).tocsc()
rhs = s['source_modes'].reshape(-1, nx*ny).T
ij = np.asarray(s['sensor_ij']); idx = ij[:,0]*ny + ij[:,1]
blocks = []
for mu in s['mu_shifts']:
    fields = splu(A + mu*eye(nx*ny, format='csc')).solve(rhs)
    blocks.append(fields[idx])
direct = np.vstack(blocks)
D, last = m._forward(s)
forward_error = np.max(abs(D-direct))
field_error = np.max(abs(last.reshape(-1, nx*ny).T-fields))
print('Direct sparse FVM versus separable solver: design / terminal modes', forward_error, field_error, flush=True)
assert max(forward_error, field_error) < 2e-12
S, y = m._profile(s, D, m._calibrate(s))
r = y-ref['r_y']; bank = np.asarray(s['row_bank_id']); z = np.asarray(s['Z_nuisance'])
print('Bank-wise profiled observation errors:', [(int(b), int((bank==b).sum()), float(np.max(abs(r[bank==b]))), float(r[bank==b].mean())) for b in np.unique(bank)])
print('Max profiled design error:', np.max(abs(S-ref['S_tilde'])))
assert np.max(abs(r)) < 5e-8
out = m.solve_beta_w02_batch_02(s)
print('Weight errors by budget:', np.max(abs(out['weights_by_budget']-ref['weights_by_budget']), axis=1))
print('Primal-dual gaps:', [c['primal_objective']-c['dual_lower_bound'] for c in out['kkt_certificates']])
