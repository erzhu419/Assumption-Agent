import importlib.util, itertools, pickle
import numpy as np
from numpy.testing import assert_allclose

path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
nx, ny = s['nx'], s['ny']

# Independently assemble physical coordinates, rather than reuse sparse Kronecker assembly.
A = np.diag(s['V_reaction_2d'].ravel(order='F'))
for j in range(ny):
    for i in range(nx):
        row = i + j * nx
        for a, p, n, stride in [('x', i, nx, 1), ('y', j, ny, nx)]:
            k, v = s['kappa_'+a], s['vel_'+a]
            h = s['L'+a] / (n-1)
            for direction in [-1, 1]:
                q = p + direction
                if not 0 <= q < n:
                    continue
                if p in (0,n-1):
                    rate = k[p]/h**2
                else:
                    rate = (k[p]+k[q])/(2*h**2) + max(-direction*v[p],0)/h
                A[row,row] += rate
                A[row,row+direction*stride] = -rate
rhs = np.stack([v.ravel(order='F') for v in s['source_modes']], axis=1)
U = [np.linalg.solve(A + shift*np.eye(nx*ny), rhs) for shift in s['mu_shifts']]
i, j = s['sensor_ij'].T
S = np.concatenate([v[i+j*nx] for v in U])
print('forward sensor max abs error:', np.max(abs(m._responses(s)[0]-S)))
assert_allclose(m._responses(s)[0], S, atol=2e-13, rtol=0)
npulse = len(item['campaign_diagnostics'])
err = []
for p in item['campaign_diagnostics']:
    y = m._calibrate(dict(s, y_obs_raw=p['y_pulse_raw']))
    err.append(np.sqrt(np.mean((S@p['w_pulse']-y).reshape(len(U),-1)**2,axis=1)))
err = np.asarray(err)
print('pulse/shift RMSE:', err.tolist())
assert err.max() < 4e-6

Z = s['Z_nuisance']
P = np.eye(len(S)) - Z @ np.linalg.solve(Z.T@Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
X, y = P @ S, P @ m._calibrate(s)
ridge, cap, total = (s[k] for k in ['source_ridge','u_box','total_budget'])
H = X.T@X + ridge*np.eye(X.shape[1]); b = -X.T@y
J = lambda w: (np.linalg.norm(X@w-y)**2 + ridge*np.dot(w,w))/2

def verify(H,b,cap,total,w,nu):
    g = H@w+b
    lower, upper = np.maximum(g-nu,0), np.maximum(nu-g,0)
    errs = [abs(w.sum()-total), max(0,-w.min(),w.max()-cap), np.max(abs(lower*w)), np.max(abs(upper*(cap-w)))]
    assert max(errs) < 1e-10, errs
    return max(errs)

out = m.solve_batch(s)
w, c = out['weights'], out['kkt_certificate']
nu = c['nu_sum']; g = H@w+b
kkt = verify(H,b,cap,total,w,nu)
assert_allclose(c['weights'],w,rtol=0,atol=0)
assert_allclose(c['lam_lower'],np.maximum(g-nu,0),rtol=0,atol=1e-12)
assert_allclose(c['lam_upper'],np.maximum(nu-g,0),rtol=0,atol=1e-12)
assert_allclose(c['primal_objective'],J(w),rtol=0,atol=1e-12)
assert_allclose(c['dual_lower_bound'],J(w)-g@w+nu*total+cap*np.minimum(0,g-nu).sum(),rtol=0,atol=1e-12)
assert abs(c['primal_objective']-c['dual_lower_bound']) < 1e-10
f = (U[-1]@w).reshape(nx,ny,order='F')
assert_allclose(out['terminal_field'],f,rtol=0,atol=2e-13)
print('terminal max abs error:', np.max(abs(out['terminal_field']-f)))
print('public KKT max residual:', kkt, 'primal-dual gap:', c['primal_objective']-c['dual_lower_bound'])

# Exercise face transitions, including integer-cap budgets where the minimizer can be a vertex.
worst = 0.0
for scale in [0.0, 1.0, 20.0]:
    for t in [0.0, 0.01*cap, cap, 3*cap, total, (len(w)-0.01)*cap, len(w)*cap]:
        ws, ns = m._box_simplex(H,b*scale,cap,t)
        worst = max(worst,verify(H,b*scale,cap,t,ws,ns))
print('worst KKT residual over 21 budget and observation slices:', worst)
