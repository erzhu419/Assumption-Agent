import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def periodic_axis(k, v, h):
    n = len(k)
    a = np.zeros((n, n))
    for i in range(n):
        j, l = (i-1) % n, (i+1) % n
        left = (k[i]+k[j])/(2*h*h) + max(v[i], 0)/h
        right = (k[i]+k[l])/(2*h*h) + max(-v[i], 0)/h
        a[i,i] += left+right
        a[i,j] -= left
        a[i,l] -= right
    return a

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for i, item in enumerate(items):
    s = item['study']
    nx, ny = s['nx'], s['ny']
    ax, ay = (periodic_axis(s['kappa_'+d], s['vel_'+d], s['L'+d]/(n-1)) for d,n in [('x',nx),('y',ny)])
    L = np.kron(np.eye(ny), ax) + np.kron(ay, np.eye(nx)) + np.diag(s['V_reaction_2d'].flatten(order='F'))
    I = np.eye(nx*ny)
    rhs = np.column_stack([mode.flatten(order='F') for mode in s['source_modes']])
    solutions = [np.linalg.solve(L + t*I, rhs) for t in s['mu_shifts']]
    ix, iy = np.asarray(s['sensor_ij']).T
    S = np.vstack([u[ix + nx*iy] for u in solutions])
    assert np.allclose(S, m._forward(s)[0], rtol=0, atol=1e-11)
    y = m._calibrate(s)
    Z = s['Z_nuisance']
    P = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    A, b = S-Z@(P@S), y-Z@(P@y)
    ridge, cap = s['source_ridge'], s['u_box']
    obj = lambda w: 0.5*(np.linalg.norm(A@w-b)**2 + ridge*(w@w))
    grad = lambda w: A.T@(A@w-b) + ridge*w
    n = S.shape[1]
    for label, budget in [('public', s['total_budget']), ('low-budget', .05*n*cap), ('near-capacity', .95*n*cap)]:
        out = m.solve_batch(dict(s, total_budget=budget))
        w, cert = np.asarray(out['weights']), out['kkt_certificate']
        lo, hi = np.asarray(cert['lam_lower']), np.asarray(cert['lam_upper'])
        g, nu = grad(w), cert['nu_sum']
        dual = obj(w)-g@w+nu*budget+cap*np.minimum(0, g-nu).sum()
        gap = obj(w)-dual
        opt = minimize(obj, np.full(n,budget/n), jac=grad, bounds=[(0,cap)]*n, constraints=[{'type':'eq','fun':lambda w: w.sum()-budget,'jac':lambda w: np.ones_like(w)}], method='SLSQP', options={'ftol':1e-13,'maxiter':500})
        feas = max(abs(w.sum()-budget), -w.min(), w.max()-cap)
        stat = np.max(np.abs(g-nu-lo+hi))
        comp = max(np.max(np.abs(lo*w)), np.max(np.abs(hi*(cap-w))))
        expected = (solutions[-1] @ w).reshape(nx, ny, order='F')
        field = np.max(np.abs(out['terminal_field']-expected))
        delta = abs(obj(w)-opt.fun)
        print(f'item={i} slice={label} feasibility={feas:.3g} stationarity={stat:.3g} complementarity={comp:.3g} gap={gap:.3g} objective_delta={delta:.3g} field_error={field:.3g}', flush=True)
        assert opt.success, opt.message
        assert feas < 1e-9 and stat < 1e-9 and comp < 1e-7 and field < 1e-10
        assert min(lo.min(), hi.min()) >= 0
        assert np.allclose(w, cert['weights'], rtol=0, atol=1e-12)
        assert abs(obj(w)-cert['primal_objective']) < 1e-10
        assert abs(dual-cert['dual_lower_bound']) < 1e-10
        assert -1e-10 <= gap < 1e-6 and delta < 1e-7
