import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

spec = importlib.util.spec_from_file_location('beta04', '/holdout_task/beta_w01_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_beta_w01_batch_04.pkl', 'rb') as f:
    cases = pickle.load(f)

def ablation(s, factor, additive):
    modes = s['source_modes']
    nmode, nx, ny = modes.shape
    ax = m._build_1d(s['Lx'], nx, factor*s['kappa_x'], s['vel_x'], s['robin_x'], s['reaction_x'])
    ay = m._build_1d(s['Ly'], ny, factor*s['kappa_y'], s['vel_y'], s['robin_y'], s['reaction_y'])
    v = s['V_reaction_2d']
    if additive:
        v = v.mean(axis=0) + v.mean(axis=1)[:, None] - v.mean()
    base = kron(ax, eye(ny), format='csc') + kron(eye(nx), ay, format='csc')
    i, j = s['sensor_ij'].T
    rhs = modes.reshape(nmode, -1).T
    return np.vstack([splu(base + diags(v.ravel() + shift, format='csc')).solve(rhs)[i*ny+j] for shift in s['mu_shifts']])

for k, case in enumerate(cases):
    s, gt = case['study'], case['gt']
    y = m._calibrated_observations(s)
    z = s['Z_nuisance']
    banks = s['row_bank_id']
    raw_v = s['V_reaction_2d']
    vx_y = raw_v.mean(axis=0) + raw_v.mean(axis=1)[:, None] - raw_v.mean()
    print('study', k, 'relative nonadditive reaction', np.linalg.norm(raw_v-vx_y)/np.linalg.norm(raw_v))
    full = m._responses(s)[0]
    budget, cap = s['total_budget'], s['u_box']
    metrics = {}
    for name, design in [('full', full), ('unscaled', ablation(s, 1., False)), ('additive', ablation(s, 1.45, True))]:
        profiled = m._profile(z, design, s['nuisance_ridge'])
        err = np.linalg.norm(profiled-gt['S_tilde'])/np.linalg.norm(gt['S_tilde'])
        rmses = []
        a = np.column_stack((design, z))
        p = design.shape[1]
        penalty = np.r_[np.full(p, s['source_ridge']), np.full(z.shape[1], s['nuisance_ridge'])]
        equality = np.r_[np.ones(p), np.zeros(z.shape[1])]
        for b in np.unique(banks):
            test = banks == b
            tr = a[~test]
            gram = tr.T @ tr + np.diag(penalty)
            rhs = tr.T @ y[~test]
            def objective(x):
                return 0.5*x @ gram @ x - rhs @ x, gram @ x - rhs
            fit = minimize(objective, np.r_[np.full(p, budget/p), np.zeros(z.shape[1])], jac=True, method='SLSQP', bounds=[(0., cap)]*p+[(None, None)]*z.shape[1], constraints=[dict(type='eq', fun=lambda x: equality @ x - budget, jac=lambda x: equality)], options=dict(ftol=1e-14, maxiter=500))
            if not fit.success:
                raise RuntimeError(fit.message)
            resid = a[test] @ fit.x - y[test]
            rmses.append((int(b), float(np.sqrt(np.mean(resid**2))), float(np.mean(resid)), float(np.std(resid))))
        metrics[name] = err
        print(name, 'profiled design relL2', err, 'held-out bank (id, RMSE, mean residual, centered RMSE)', rmses)
    assert metrics['full'] < 1e-10 and metrics['full'] < min(metrics['unscaled'], metrics['additive']), metrics
