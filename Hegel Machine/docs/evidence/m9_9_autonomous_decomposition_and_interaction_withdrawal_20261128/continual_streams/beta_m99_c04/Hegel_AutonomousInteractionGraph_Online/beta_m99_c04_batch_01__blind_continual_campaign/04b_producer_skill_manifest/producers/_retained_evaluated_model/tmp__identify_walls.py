import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

spec = importlib.util.spec_from_file_location('prior', '/tmp/hegel_continual_archive/beta_m99_c04_s00_solver.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
nx, ny = s['nx'], s['ny']
axes = []
for a in 'xy':
    k = s['kappa_'+a]; h = s['L'+a] / (len(k)-1); r = s['robin_'+a]
    base = prior._axis(k, s['vel_'+a], h, r).toarray()
    axes.append((base, k / h**2, r / h))

def axis(entry, p):
    base, diffusion, robin = entry
    A = base.copy()
    for i, j, d, leak in ((0, 1, p[0], p[2]), (-1, -2, p[1], p[3])):
        A[i, j] = -d * diffusion[i]
        A[i, i] = d * diffusion[i] + leak * robin
    return csc_matrix(A)

I = eye(nx*ny, format='csc')
V = diags(s['V_reaction_2d'].ravel(order='F'), format='csc')
ix, iy = s['sensor_ij'].T
idx = ix + nx * iy
pulses = item['campaign_diagnostics']
W = np.stack([p['w_pulse'] for p in pulses], axis=1)
F = np.stack([a.ravel(order='F') for a in s['source_modes']], axis=1) @ W
Y = np.stack([prior._calibrate(dict(s, y_obs_raw=p['y_pulse_raw'])) for p in pulses], axis=1)
Q = np.linalg.qr(s['Z_nuisance'], mode='reduced')[0]

def residual(p):
    A = kron(eye(ny), axis(axes[0], p[:4]), format='csc') + kron(axis(axes[1], p[4:]), eye(nx), format='csc') + V
    R = np.vstack([splu(A + mu*I).solve(F)[idx] for mu in s['mu_shifts']]) - Y
    return R - Q @ (Q.T @ R)

def score(p):
    return np.sqrt(np.mean(residual(p).reshape(len(s['mu_shifts']), len(idx), -1)**2, axis=1)).round(8).tolist()

for name, p in [('open', [1,1,1,1]*2), ('sealed', [1,1,0,0]*2), ('reflected', [2,2,0,0]*2), ('sealed_x', [1,1,0,0,1,1,1,1]), ('sealed_y', [1,1,1,1,1,1,0,0])]:
    print(name, score(p))
for held in [None, *range(len(pulses))]:
    fit_cols = [j for j in range(len(pulses)) if j != held]
    fit = least_squares(lambda p: residual(p)[:,fit_cols].ravel(), np.ones(8), bounds=(0, 5), gtol=1e-13, ftol=1e-13, xtol=1e-13)
    print('held', held, 'p', fit.x.round(6).tolist(), 'RMS by shift/pulse', score(fit.x))
for held in range(len(pulses)):
    fit_cols = [j for j in range(len(pulses)) if j != held]
    x = least_squares(lambda p: residual([1,1,p[0],p[0],1,1,p[1],p[1]])[:,fit_cols].ravel(), [.5,.5], bounds=(0, 5), gtol=1e-13).x
    print('Robin-only held', held, 'coeffs', x.round(7).tolist(), 'RMS by shift/pulse', score([1,1,x[0],x[0],1,1,x[1],x[1]]))
