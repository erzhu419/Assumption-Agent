import json, pickle, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import solve_downstream, _box_simplex
from _verified_upstream import calibrate_upstream

root = '/holdout_task/'
items = pickle.load(open(root+'public_batch.pkl', 'rb'))
payload = json.load(open(root+'_upstream_param_payload.json'))
for item in items:
    raw = item['study']
    s = calibrate_upstream(raw, payload)
    out = solve_batch(raw)
    direct = solve_downstream(s)
    assert out['kkt_certificate'] == direct['kkt_certificate']
    np.testing.assert_array_equal(out['terminal_field'], direct['terminal_field'])
    dims = (s['nx'], s['ny'])
    size = np.prod(dims)
    A = np.diag(s['V_reaction_2d'].ravel(order='F').copy())
    for j in range(dims[1]):
        for i in range(dims[0]):
            row = i + dims[0]*j
            for a, name in enumerate('xy'):
                n, q = dims[a], (i, j)[a]
                h = s['L'+name]/(n-1)
                k, v = s['kappa_'+name], s['vel_'+name][q]
                stride = (1, dims[0])[a]
                for sign in (-1, 1):
                    t = q+sign
                    if a == 1 and q in (0, n-1):
                        if not 0 <= t < n:
                            continue
                        rate = k[q]/h**2
                    else:
                        t %= n
                        rate = (k[q]+k[t])/(2*h**2)+max(-sign*v, 0)/h
                    A[row,row] += rate
                    A[row,row+(t-q)*stride] -= rate
    modes = np.column_stack([m.ravel(order='F') for m in s['source_modes']])
    ix, iy = s['sensor_ij'].T
    blocks = []
    for mu in s['mu_shifts']:
        M = A + mu*np.eye(size)
        U = np.linalg.solve(M, modes)
        blocks.append(U[ix+dims[0]*iy])
    S, z, y = np.vstack(blocks), s['Z_nuisance'], s['y_obs_raw']
    P = np.eye(len(y))-z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
    X, yy = P@S, P@y
    ridge, cap, total = (s[k] for k in ('source_ridge','u_box','total_budget'))
    def obj(w):
        r = X@w-yy
        return (r@r+ridge*(w@w))/2
    def grad(w):
        return X.T@(X@w-yy)+ridge*w
    w, c = out['weights'], out['kkt_certificate']
    lo, hi = map(np.asarray, (c['lam_lower'],c['lam_upper']))
    np.testing.assert_array_equal(w,c['weights'])
    assert abs(w.sum()-total)<1e-12 and w.min()>=0 and w.max()<=cap
    assert lo.min()>=0 and hi.min()>=0
    stationarity = np.max(abs(grad(w)-c['nu_sum']-lo+hi))
    comp = max(np.max(abs(lo*w)),np.max(abs(hi*(cap-w))))
    gap = obj(w)-c['dual_lower_bound']
    assert stationarity<1e-10 and comp<1e-10 and abs(gap)<1e-10
    assert abs(c['primal_objective']-obj(w))<1e-10
    g = grad(w)
    assert abs(c['dual_lower_bound']-(obj(w)-g@w+c['nu_sum']*total+cap*np.minimum(0,g-c['nu_sum']).sum()))<1e-10
    ref = minimize(obj,np.full(len(w),total/len(w)),jac=grad,method='SLSQP',bounds=[(0,cap)]*len(w),constraints=[{'type':'eq','fun':lambda v:v.sum()-total,'jac':lambda v:np.ones_like(v)}],options={'ftol':1e-13})
    assert ref.success and abs(obj(w)-ref.fun)<1e-10
    field = out['terminal_field'].ravel(order='F')
    field_residual = np.max(abs(M@field-modes@w))
    np.testing.assert_allclose(field,U@w,atol=1e-11,rtol=1e-11)
    assert field_residual<1e-10
    print(f'public: stationarity={stationarity:.3g}, complementarity={comp:.3g}, dual gap={gap:.3g}, field residual={field_residual:.3g}, optimizer objective delta={abs(obj(w)-ref.fun):.3g}')

rng=np.random.default_rng(61)
worst=0.
for _ in range(40):
    B=rng.normal(size=(10,8)); H=B.T@B+0.04*np.eye(8)
    q=rng.normal(size=8)*10; cap=0.42
    for total in (0., .42, .84, 1.4, 3.1, 8*cap):
        w,nu=_box_simplex(H,q,cap,total)
        g=H@w+q-nu
        err=max(abs(w.sum()-total),max(0,-w.min(),w.max()-cap),np.max(abs(np.maximum(g,0)*w)),np.max(abs(np.minimum(g,0)*(cap-w))))
        worst=max(worst,err)
assert worst<1e-9
print(f'240 box-simplex stress cases: worst KKT residual={worst:.3g}')
