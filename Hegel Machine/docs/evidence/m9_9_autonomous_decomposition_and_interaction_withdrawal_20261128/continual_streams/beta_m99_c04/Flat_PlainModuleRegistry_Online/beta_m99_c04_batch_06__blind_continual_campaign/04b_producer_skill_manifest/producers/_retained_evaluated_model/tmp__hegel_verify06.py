import sys, pickle, json
import numpy as np
from itertools import product
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream, _box_simplex
from _verified_upstream import calibrate_upstream
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    raw = pickle.load(f)[0]['study']
with open('/holdout_task/_upstream_param_payload.json') as f:
    s = calibrate_upstream(raw, json.load(f))
out = solve_batch(raw)
direct = solve_downstream(s)
assert out['kkt_certificate'] == direct['kkt_certificate']
np.testing.assert_array_equal(out['terminal_field'], direct['terminal_field'])
nx, ny = s['nx'], s['ny']
L = np.diag(s['V_reaction_2d'].ravel(order='F'))
for j, i in product(range(ny), range(nx)):
    r = i + nx*j
    for d, a, n, stride in [('x', i, nx, 1), ('y', j, ny, nx)]:
        h = s['L'+d]/(n-1)
        k, v = s['kappa_'+d], s['vel_'+d][a]
        for direction in [-1, 1]:
            b = (a+direction) % n
            rate = (k[a]+k[b])/(2*h*h)+max(0., -direction*v)/h
            L[r,r] += rate
            L[r,r+(b-a)*stride] -= rate
rhs = np.array([m.ravel(order='F') for m in s['source_modes']]).T
sensor_ids = s['sensor_ij'] @ np.array([1,nx])
blocks = []
for mu in s['mu_shifts']:
    M = L + mu*np.eye(nx*ny)
    U = np.linalg.solve(M, rhs)
    blocks.append(U[sensor_ids])
Z = s['Z_nuisance']
P = np.eye(len(Z)) - Z @ np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
A, y = P@np.vstack(blocks), P@s['y_obs_raw']
w, c = out['weights'], out['kkt_certificate']
cap, budget, ridge = s['u_box'], s['total_budget'], s['source_ridge']
r = A@w-y
g = A.T@r+ridge*w
obj = (r@r+ridge*(w@w))/2
lo, hi = np.array(c['lam_lower']), np.array(c['lam_upper'])
nu = c['nu_sum']
stationarity = max(abs(g-nu-lo+hi))
complementarity = max(max(abs(w*lo)), max(abs((cap-w)*hi)))
field_residual = max(abs(M@out['terminal_field'].ravel(order='F')-rhs@w))
assert np.min(w)>=0 and np.max(w)<=cap and abs(w.sum()-budget)<1e-12
assert min(lo.min(),hi.min())>=0
np.testing.assert_allclose(c['weights'], w, atol=0, rtol=0)
np.testing.assert_allclose(c['primal_objective'], obj, atol=1e-12, rtol=0)
np.testing.assert_allclose(c['dual_lower_bound'], obj-g@w+nu*budget+cap*np.minimum(0.,g-nu).sum(), atol=1e-12, rtol=0)
assert max(stationarity,complementarity,field_residual,abs(obj-c['dual_lower_bound'])) < 1e-10
worst = 0.
for budget in (0., cap, 2*cap, 3.33*cap, len(w)*cap):
    for scale in (1., 100.):
        H = A.T@A+ridge*np.eye(len(w)); b = scale*(A.T@y)
        x, p = _box_simplex(H,b,cap,budget)
        gx = H@x-b
        gap = gx@x-p*budget-cap*np.minimum(0.,gx-p).sum()
        err = max(abs(gap),abs(x.sum()-budget),max(0.,-x.min(),x.max()-cap))
        worst = max(worst,err)
        assert err < 1e-10
print('independent dense stencil: field residual %.3g, stationarity %.3g, complementarity %.3g, public gap %.3g, stress worst %.3g' % (field_residual, stationarity, complementarity, obj-c['dual_lower_bound'], worst))
