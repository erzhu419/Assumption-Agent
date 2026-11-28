import importlib.util
import pickle
import numpy as np

path = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', path + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open(path + 'public_batch.pkl', 'rb'))[0]['study']
out = m.solve_batch(s)
w, u, cert = (out[k] for k in ('weights', 'terminal_field', 'kkt_certificate'))
nx, ny = s['nx'], s['ny']
def row_op(a, n):
    h = s['L' + a] / (n - 1)
    k, v = s['kappa_' + a], s['vel_' + a]
    op = np.zeros((n, n))
    for i in range(n):
        l, r = (i-1) % n, (i+1) % n
        west = (k[i] + k[l]) / (2*h*h) + max(v[i], 0)/h
        east = (k[i] + k[r]) / (2*h*h) + max(-v[i], 0)/h
        op[i,l] -= west
        op[i,r] -= east
        op[i,i] += west + east
    assert np.max(np.abs(op.sum(axis=1))) < 1e-10
    return op
ax, ay = row_op('x', nx), row_op('y', ny)
V = s['V_reaction_2d']
mu = s['mu_shifts'][-1]
source = np.tensordot(w, s['source_modes'], axes=1)
field_res = np.max(np.abs(ax @ u + u @ ay.T + (V+mu)*u - source))
assert field_res < 1e-10
L = np.kron(np.eye(ny), ax) + np.kron(ay, np.eye(nx)) + np.diag(V.ravel(order='F'))
rhs = np.stack([q.ravel(order='F') for q in s['source_modes']], axis=1)
i, j = s['sensor_ij'].T
S = np.vstack([np.linalg.solve(L + shift*np.eye(nx*ny), rhs)[i+nx*j] for shift in s['mu_shifts']])
y = m._calibrate(s)
Z = s['Z_nuisance']
P = np.eye(len(y)) - Z @ np.linalg.solve(Z.T@Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
err = P @ (S@w-y)
g = (P@S).T @ err + s['source_ridge']*w
lo, hi, nu = (np.asarray(cert[k]) for k in ('lam_lower', 'lam_upper', 'nu_sum'))
cap, budget = s['u_box'], s['total_budget']
J = (err@err + s['source_ridge']*(w@w))/2
D = J-g@w+nu*budget+cap*np.minimum(g-nu,0).sum()
checks = {'budget':abs(w.sum()-budget), 'stationarity':np.max(np.abs(g-nu-lo+hi)), 'lower_complementarity':np.max(np.abs(lo*w)), 'upper_complementarity':np.max(np.abs(hi*(cap-w))), 'objective_error':abs(cert['primal_objective']-J), 'bound_error':abs(cert['dual_lower_bound']-D), 'duality_gap':abs(J-D)}
assert np.all((w >= 0) & (w <= cap)) and np.all(lo >= 0) and np.all(hi >= 0)
assert np.array_equal(w, cert['weights']) and max(checks.values()) < 1e-10
print('field PDE infinity residual:', field_res, 'certificate errors:', checks)
# Hold out each bank across all shifts; profile nuisance using only retained rows.
bank = np.asarray(s['row_bank_id']).ravel()
assert len(y) % len(bank) == 0
bank = np.tile(bank, len(y)//len(bank))
score = {}
for b in np.unique(bank):
    t = bank != b
    X, z = S[t], Z[t]
    p = np.linalg.solve(z.T@z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T)
    A, r = X-z@(p@X), y[t]-z@(p@y[t])
    wt, _ = m._box_simplex(A.T@A+s['source_ridge']*np.eye(len(w)), -A.T@r, cap, budget)
    et = S[~t]@wt + Z[~t]@(p@(y[t]-X@wt)) - y[~t]
    score[int(b)] = float(np.linalg.norm(et)/np.sqrt(len(et)))
print('held-out bank RMSE:', score)
assert max(score.values()) < 0.025
