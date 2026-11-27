import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as batch
import downstream_solver as downstream


def dense_axis(kap, vel, h, wrap):
    n = len(kap)
    a = np.zeros((n, n))
    for i in range(n):
        if not wrap and i in (0, n-1):
            a[i, i] = kap[i] / h**2
            a[i, 1 if i == 0 else n-2] = -a[i, i]
            continue
        j, k = (i-1) % n, (i+1) % n
        a[i, j] = -(kap[i]+kap[j])/(2*h**2) - max(vel[i], 0)/h
        a[i, k] = -(kap[i]+kap[k])/(2*h**2) - max(-vel[i], 0)/h
        a[i, i] = -a[i, j] - a[i, k]
    return a


def reference(s):
    nx, ny = s['nx'], s['ny']
    ax, ay = [dense_axis(s['kappa_'+q], s['vel_'+q], s['L'+q]/(n-1), q == 'x') for q,n in [('x',nx),('y',ny)]]
    I = np.eye(nx*ny)
    L = np.kron(np.eye(ny), ax) + np.kron(ay, np.eye(nx)) + np.diag(s['V_reaction_2d'].flatten('F'))
    rhs = np.stack([m.flatten('F') for m in s['source_modes']], axis=1)
    U = [np.linalg.solve(L+mu*I, rhs) for mu in s['mu_shifts']]
    ij = s['sensor_ij']
    return np.concatenate([u[ij[:,0]+nx*ij[:,1],:] for u in U]), U[-1]


def audit(s, out, S, U, label):
    w, cert = out['weights'], out['kkt_certificate']
    cap, budget, ridge = [s[k] for k in ['u_box','total_budget','source_ridge']]
    Z = s['Z_nuisance']
    M = np.eye(len(Z)) - Z @ np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
    r = M @ (S@w-s['y_obs_raw'])
    J = (r@r+ridge*(w@w))/2
    g = (M@S).T@r+ridge*w
    nu = cert['nu_sum']
    lo, hi = np.array(cert['lam_lower']), np.array(cert['lam_upper'])
    dual = J-g@w+nu*budget+cap*np.minimum(g-nu,0).sum()
    feasibility = max(abs(w.sum()-budget), -w.min(), w.max()-cap, 0.)
    stationarity = np.max(np.abs(g-nu-lo+hi))
    comp = max(np.max(np.abs(w*lo)), np.max(np.abs((cap-w)*hi)))
    terminal = np.max(np.abs(out['terminal_field']-(U@w).reshape((s['nx'],s['ny']),order='F')))
    errors = [feasibility, stationarity, comp, abs(J-dual), abs(J-cert['primal_objective']), abs(dual-cert['dual_lower_bound']),terminal]
    np.testing.assert_array_equal(cert['weights'], w)
    assert np.all(lo>=0) and np.all(hi>=0)
    assert max(errors)<1e-9, (label,errors)
    print(label,'max_error',max(errors),'active_lower',sum(w<=1e-10),'active_upper',sum(w>=cap-1e-10))

with open('/holdout_task/public_batch.pkl','rb') as f:
    items = pickle.load(f)
for i,item in enumerate(items):
    raw = item['study']
    s = batch._up.calibrate_upstream(raw, batch._PAY)
    S,U = reference(s)
    audit(s,batch.solve_batch(raw),S,U,f'public {i}')
    n = S.shape[1]
    for j,budget in enumerate((0.,s['u_box']*n,2*s['u_box'],s['total_budget'])):
        data = dict(s,total_budget=budget,y_obs_raw=s['y_obs_raw']+np.linspace(-1.,1.,len(S)))
        audit(data,downstream.solve_downstream(data),S,U,f'constraint slice {i}/{j}')
