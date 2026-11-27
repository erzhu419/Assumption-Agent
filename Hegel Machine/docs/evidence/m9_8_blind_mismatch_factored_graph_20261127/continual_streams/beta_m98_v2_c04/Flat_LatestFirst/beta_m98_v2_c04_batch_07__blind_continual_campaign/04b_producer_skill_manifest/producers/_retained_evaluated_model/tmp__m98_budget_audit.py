import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
S, U = m._forward(s)
Z = s['Z_nuisance']
R = np.eye(len(Z)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
A = R @ S
rho = s['source_ridge']
n = S.shape[1]
cap = s['u_box']
worst = 0.
worst_oracle = 0.
for label, raw in [('observed', s['y_obs_raw'])]:
    y = R @ m._calibrate(dict(s, y_obs_raw=raw))
    def objective(w):
        r = A @ w - y
        return .5 * (r @ r + rho * (w @ w))
    def grad(w):
        return A.T @ (A @ w - y) + rho * w
    for budget in [0., .01*cap, cap, s['total_budget'], (n-1)*cap, (n-.01)*cap, n*cap]:
        out = m.solve_batch(dict(s, y_obs_raw=raw, total_budget=budget))
        w = np.array(out['weights'])
        cert = out['kkt_certificate']
        nu = cert['nu_sum']
        lo, hi = np.array(cert['lam_lower']), np.array(cert['lam_upper'])
        g = grad(w)
        J = objective(w)
        dual = J - g@w + nu*budget + cap*np.minimum(0, g-nu).sum()
        err = max(abs(w.sum()-budget), -w.min(), w.max()-cap, -lo.min(), -hi.min(), np.max(abs(g-nu-lo+hi)), np.max(abs(lo*w)), np.max(abs(hi*(cap-w))), abs(dual-J), abs(cert['dual_lower_bound']-dual), abs(cert['primal_objective']-J), np.max(abs(np.array(cert['weights'])-w)), np.max(abs(out['terminal_field']-np.einsum('m,mij->ij',w,U))))
        worst = max(worst, err)
        if 0 < budget < n*cap:
            opt = minimize(objective, np.full(n,budget/n), jac=grad, method='SLSQP', bounds=[(0,cap)]*n, constraints={'type':'eq','fun':lambda v:v.sum()-budget,'jac':lambda v:np.ones(n)}, options={'ftol':1e-13,'maxiter':500})
            assert opt.success, opt.message
            worst_oracle = max(worst_oracle, abs(J-opt.fun))
        print(f'{label} budget={budget:.8g} lower={sum(w<1e-10)} upper={sum(w>cap-1e-10)} kkt={err:.3g}')
assert worst < 1e-8 and worst_oracle < 1e-9, (worst,worst_oracle)
print('Worst budget-slice KKT error:',worst,'; worst SLSQP objective disagreement:',worst_oracle)
