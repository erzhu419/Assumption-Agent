import sys
sys.path[:0] = ['/tmp', '/holdout_task']
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from probe_boundary import s, W, Y
from downstream_solver import _axis_operator, _forward

nx, ny = s['nx'], s['ny']
axes = [_axis_operator(s['kappa_'+a], s['vel_'+a], s['L'+a]) for a in 'xy']
i, j = s['sensor_ij'].T
modes = np.column_stack([u.ravel(order='F') for u in s['source_modes']])
reaction = diags(s['V_reaction_2d'].ravel(order='F'))
identity = eye(nx*ny, format='csc')
S, _ = _forward(s)
models = {'periodic': S @ W}
for axis, label in enumerate('xy'):
    for end in (0, -1):
        ops = axes.copy()
        a = ops[axis].tolil()
        row, opposite = (0, a.shape[0]-1) if end == 0 else (a.shape[0]-1, 0)
        a[row,row] += a[row,opposite]
        a[row,opposite] = 0
        assert np.max(np.abs(np.asarray(a.sum(axis=1)))) < 1e-12
        ops[axis] = a.tocsc()
        L = (kron(eye(ny),ops[0])+kron(ops[1],eye(nx))+reaction).tocsc()
        models[f'no_{label}_wrap_row_{row}'] = np.concatenate([splu(L+mu*identity).solve(modes @ W)[i+nx*j] for mu in s['mu_shifts']])
nshift, m, npulse = len(s['mu_shifts']), len(i), W.shape[1]
errors = {name:(y-Y).reshape(nshift,m,npulse) for name,y in models.items()}
worst, min_margin, wins = 0., np.inf, 0
for shift in range(nshift):
    for pulse in range(npulse):
        train_shift = np.arange(nshift) != shift
        train_pulse = np.arange(npulse) != pulse
        scores = sorted((np.sqrt(np.mean(r[train_shift][:,:,train_pulse]**2)), name) for name,r in errors.items())
        e, name = scores[0]
        wins += name == 'periodic'
        worst = max(worst, float(np.sqrt(np.mean(errors[name][shift,:,pulse]**2))))
        min_margin = min(min_margin, scores[1][0]/e)
print('Single-link ablation worst pulse-by-shift RMS:', {name:float(np.sqrt(np.mean(r*r,axis=1)).max()) for name,r in errors.items()})
print(f'ng_t04: full periodic won {wins}/{nshift*npulse} folds against every individual wrap-link deletion; worst held-out RMS={worst:.8g}; minimum runner-up/winner training RMS={min_margin:.4g}')
assert wins == nshift*npulse and worst < 1e-5
