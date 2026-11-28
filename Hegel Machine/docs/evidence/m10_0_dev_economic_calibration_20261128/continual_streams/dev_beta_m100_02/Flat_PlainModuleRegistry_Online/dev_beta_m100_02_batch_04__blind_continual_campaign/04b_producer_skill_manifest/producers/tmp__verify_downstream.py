import itertools
import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _box_simplex, solve_downstream
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
cal = calibrate_upstream(dict(s, reference_tracer_pulses=item['campaign_diagnostics']))
nx, ny = s['nx'], s['ny']

def axis(name):
    k, v = s['kappa_'+name], s['vel_'+name]
    n = len(k)
    h = s['L'+name]/(n-1)
    M = np.zeros((n,n))
    for i in range(n):
        if name == 'y' and (i == 0 or i == n-1):
            M[i,1 if i == 0 else n-2] = -k[i]/h**2
        else:
            for step in (-1,1):
                j = (i+step)%n
                M[i,j] -= (k[i]+k[j])/(2*h*h) + max(0,-step*v[i])/h
        M[i,i] = -M[i].sum()
    return M

L = np.kron(np.eye(ny),axis('x')) + np.kron(axis('y'),np.eye(nx)) + np.diag(s['V_reaction_2d'].flatten(order='F'))
rhs = np.column_stack([mode.flatten(order='F') for mode in s['source_modes']])
idx = np.array([i+nx*j for i,j in s['sensor_ij']])
U = [np.linalg.solve(L+shift*np.eye(nx*ny), rhs) for shift in s['mu_shifts']]
S = np.concatenate([a[idx] for a in U])
pulses = cal['reference_tracer_pulses']
E = np.stack([S @ p['w_pulse'] - p['y_pulse_raw'] for p in pulses])
rmse = np.sqrt(np.mean(E.reshape(len(pulses),len(U),len(idx))**2,axis=2))
assert rmse.max() < 3e-6, rmse
print('Pulse × shift RMSE:', rmse.tolist(), 'worst:', rmse.max())
Z = s['Z_nuisance']; Q = np.eye(len(Z))-Z @ np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
A, b = Q@S, Q@cal['y_obs_raw'].ravel()
H = A.T@A+s['source_ridge']*np.eye(S.shape[1]); c = -A.T@b
u = s['u_box']; total = s['total_budget']; n = len(c)

def exhaustive(H,c,u,total):
    n = len(c)
    best = np.inf
    for state in itertools.product((-1,0,1),repeat=n):
        state = np.asarray(state)
        F = np.flatnonzero(state==0)
        w = np.where(state==1,u,0.)
        if len(F):
            h = H[np.ix_(F,F)]
            mat = np.block([[h,-np.ones((len(F),1))],[np.ones((1,len(F))),np.zeros((1,1))]])
            w[F] = np.linalg.solve(mat,np.r_[-c[F]-(H@w)[F],total-w.sum()])[:-1]
        if abs(w.sum()-total)<1e-10 and w.min()>=-1e-10 and w.max()<=u+1e-10:
            value = w@H@w/2+c@w
            if value<best:
                best, answer = value, w
    return answer

out = solve_batch(s)
w = out['weights']; cert = out['kkt_certificate']; nu = cert['nu_sum']
w_ref = exhaustive(H,c,u,total)
np.testing.assert_allclose(w, w_ref, rtol=0, atol=1e-10)
np.testing.assert_allclose(solve_downstream(cal)['weights'],w,rtol=0,atol=1e-12)
np.testing.assert_allclose(cert['weights'],w,rtol=0,atol=0)
field_error = np.max(np.abs(out['terminal_field']-(U[-1]@w_ref).reshape(nx,ny,order='F')))
assert field_error < 1e-10
lo, hi = np.asarray(cert['lam_lower']), np.asarray(cert['lam_upper'])
g = H@w+c
J = (np.linalg.norm(A@w-b)**2+s['source_ridge']*(w@w))/2
D = J-g@w+nu*total+u*np.minimum(0,g-nu).sum()
assert np.all(lo>=0) and np.all(hi>=0)
errors = dict(budget=abs(w.sum()-total), field=field_error, weights=np.max(abs(w-w_ref)), stationarity=np.max(abs(g-nu-lo+hi)), lower_complementarity=np.max(abs(lo*w)), upper_complementarity=np.max(abs(hi*(u-w))), objective=abs(J-cert['primal_objective']), dual=abs(D-cert['dual_lower_bound']), gap=abs(J-D))
assert max(errors.values())<1e-10, errors
print('Public verification errors:', errors)

rng = np.random.default_rng(9821)
maxerr = 0.
for t in range(48):
    G = rng.normal(size=(6,6)); H=G.T@G+0.04*np.eye(6)
    c = rng.normal(size=6)*[0.01,1,50][t%3]; u=0.42
    total = [0.,0.13,u,3*u,6*u-0.015,6*u][(t//3)%6]
    w,nu = _box_simplex(H,c,u,total); reduced=H@w+c-nu
    gap = reduced@w-u*np.minimum(0,reduced).sum()
    e=max(abs(w.sum()-total),max(0.,-w.min(),w.max()-u),abs(gap),np.max(abs(w-exhaustive(H,c,u,total))))
    maxerr=max(maxerr,e)
assert maxerr < 1e-9, maxerr
print('48 box-simplex stress cases, including zero/full and vertex budgets; worst absolute discrepancy:',maxerr)
