import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
A, last = m._forward(s)
for p in item['campaign_diagnostics']:
    y = m._calibrate(dict(s, y_obs_raw=p['y_pulse_raw']))
    rmse = np.sqrt(np.mean((A @ p['w_pulse'] - y).reshape(len(s['mu_shifts']), -1)**2, axis=1))
    print('pulse RMSE by shift:', rmse)
    assert rmse.max() < 2e-6

r = m.solve_batch(s)
w, u, c = (r[k] for k in ('weights', 'terminal_field', 'kkt_certificate'))
cap, total, ridge = (s[k] for k in ('u_box', 'total_budget', 'source_ridge'))
Z = s['Z_nuisance']
y = m._calibrate(s)
project = lambda d: d-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@d)
D, b = project(A), project(y)
g = D.T@(D@w-b)+ridge*w
J = lambda v: .5*(np.linalg.norm(D@v-b)**2+ridge*(v@v))
lo, hi, nu = (np.asarray(c[k]) for k in ('lam_lower', 'lam_upper', 'nu_sum'))
errors = [max(0.,-w.min(),w.max()-cap), abs(w.sum()-total), np.max(abs(np.asarray(c['weights'])-w)), np.max(abs(g-nu-lo+hi)), np.max(abs(lo*w)), np.max(abs(hi*(cap-w))), abs(c['primal_objective']-J(w)), abs(J(w)-c['dual_lower_bound'])]
assert min(lo.min(), hi.min()) >= 0 and max(errors) < 1e-9, errors

# Independent stencil application, including seams and corners, checks field ordering as well.
Lu = s['V_reaction_2d']*u + s['mu_shifts'][-1]*u
for a, ax in enumerate('xy'):
    k, v = s['kappa_'+ax], s['vel_'+ax]
    h = s['L'+ax]/(len(k)-1)
    shape = (-1, 1) if a == 0 else (1, -1)
    for direction in (-1, 1):
        rate = ((k+np.roll(k,-direction))/(2*h*h)+np.maximum(-direction*v,0)/h).reshape(shape)
        Lu += rate*(u-np.roll(u,-direction,axis=a))
field_error = np.max(abs(Lu-np.tensordot(w,s['source_modes'],axes=1)))
assert field_error < 1e-8, field_error

# Brute-force box faces independently certify the optimizer on the actual design.
H, q = D.T@D+ridge*np.eye(w.size), D.T@b
worst = 0.
for budget in (total, 0., cap, cap*(w.size-.35), cap*w.size):
    wc, nc = m._box_budget_qp(H,q,cap,budget)
    best = np.inf
    for pattern in itertools.product((-1,0,1),repeat=w.size):
        state = np.asarray(pattern)
        v = np.where(state == 1, cap, 0.)
        free = np.flatnonzero(state == 0)
        left = budget-v.sum()
        if left < -1e-12 or left > cap*len(free)+1e-12:
            continue
        if free.size:
            n = free.size
            M = np.ones((n+1,n+1))
            M[:n,:n] = H[np.ix_(free,free)]
            M[-1,-1] = 0
            v[free] = np.linalg.solve(M,np.r_[q[free]-H[free]@v,left])[:-1]
        if v.min() >= -1e-10 and v.max() <= cap+1e-10:
            best = min(best, J(v))
    err = max(abs(J(wc)-best),abs(wc.sum()-budget),-wc.min(),wc.max()-cap)
    assert err < 1e-9, (budget,err)
    worst = max(worst,err)
print('max certificate error:', max(errors), 'max field equation error:', field_error, 'max enumerated QP error:', worst)
