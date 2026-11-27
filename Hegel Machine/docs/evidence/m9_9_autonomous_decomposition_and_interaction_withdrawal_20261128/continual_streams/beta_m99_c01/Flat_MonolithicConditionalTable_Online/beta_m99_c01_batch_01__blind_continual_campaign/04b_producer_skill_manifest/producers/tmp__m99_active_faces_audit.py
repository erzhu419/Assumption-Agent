import pickle
import runpy
import numpy as np
from scipy.optimize import minimize

m = runpy.run_path('/holdout_task/batch_solver.py')
items = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))
worst = np.zeros(5)
for item in items:
    s = item['study']
    S, _ = m['_forward'](s)
    y = m['_calibrate'](s)
    Z = s['Z_nuisance']
    Pz = np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
    T, r = S - Z @ (Pz @ S), y - Z @ (Pz @ y)
    ridge = s['source_ridge']
    n = S.shape[1]
    H, c = T.T @ T + ridge * np.eye(n), -T.T @ r
    obj = lambda w: 0.5 * np.sum((T @ w - r)**2) + 0.5 * ridge * (w @ w)
    jac = lambda w: H @ w + c
    cap, nominal = s['u_box'], s['total_budget']
    for budget in sorted({0., 0.10 * cap, 0.5 * cap, cap, nominal, (n-0.5) * cap, (n-0.10) * cap, n*cap}):
        out = m['solve_batch'](dict(s, total_budget=budget))
        w = np.asarray(out['weights'])
        cert = out['kkt_certificate']
        nu = cert['nu_sum']
        lo, hi = map(np.asarray, (cert['lam_lower'], cert['lam_upper']))
        g, J = jac(w), obj(w)
        D = J - g @ w + nu * budget + cap * np.minimum(0, g-nu).sum()
        ref = minimize(obj, np.full(n, budget/n), jac=jac, bounds=[(0.,cap)]*n,
                       constraints=[{'type':'eq','fun':lambda v: v.sum()-budget,'jac':lambda v: np.ones(n)}],
                       method='SLSQP', options={'ftol':1e-13,'maxiter':500})
        assert ref.success, (budget, ref.message)
        errors = np.array([max(abs(w.sum()-budget), -w.min(), w.max()-cap),
                           max(np.max(abs(g-nu-lo+hi)), -lo.min(), -hi.min()),
                           max(np.max(abs(lo*w)), np.max(abs(hi*(cap-w)))),
                           max(abs(J-D), abs(J-cert['primal_objective']), abs(D-cert['dual_lower_bound'])),
                           abs(J-ref.fun)])
        print('budget', budget, 'at lower', np.count_nonzero(w < 1e-9),
              'at upper', np.count_nonzero(w > cap-1e-9), 'errors', errors.tolist())
        worst = np.maximum(worst, errors)
print('worst errors (feasibility, stationarity, complementarity, certificate, independent objective):', worst.tolist())
assert np.all(worst < [1e-10,1e-10,1e-10,1e-10,1e-9]), worst
