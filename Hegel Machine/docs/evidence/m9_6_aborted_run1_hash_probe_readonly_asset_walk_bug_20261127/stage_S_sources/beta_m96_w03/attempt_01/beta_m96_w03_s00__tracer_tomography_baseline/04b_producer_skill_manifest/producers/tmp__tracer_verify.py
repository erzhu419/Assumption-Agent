import copy, importlib.util, json, pickle, time
import numpy as np
from scipy.optimize import minimize

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'beta_m96_w03_s00_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(root+'public_beta_m96_w03_s00.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s, gt = item['study'], item['gt']
with open(root+'validate_public_schema.py') as f:
    ns = {}
    exec(f.read().split('\n_d =')[0], ns)
verify = ns['verify_box_simplex_kkt_certificate']

def check(st):
    out = mod.solve_beta_m96_w03_s00(st)
    w_err = np.max(abs(out['weights'] - gt['w_opt']))
    f_err = np.linalg.norm(out['terminal_field'] - gt['terminal_field_A']) / np.linalg.norm(gt['terminal_field_A'])
    proof = verify(gt['S_tilde'], gt['r_y'], st['source_ridge'], st['u_box'], st['total_budget'], out['kkt_certificate'], gap_tol=1e-7)
    assert proof['valid'] and w_err < 1e-6 and f_err < 1e-6, (proof, w_err, f_err)
    return dict(weight_max_abs=float(w_err), field_rel_l2=float(f_err), verified_gap=proof['gap'])

print('public', json.dumps(check(s)))
D, _ = mod._transport(s)
Z = s['Z_nuisance']
D -= Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T@D)
err = (D - gt['S_tilde']).reshape(len(s['mu_shifts']), -1, D.shape[1])
ref = gt['S_tilde'].reshape(err.shape)
shift_error = np.max(np.linalg.norm(err, axis=(1,2))/np.linalg.norm(ref, axis=(1,2)))
sensor_error = np.max(np.linalg.norm(err, axis=(0,2))/np.linalg.norm(ref, axis=(0,2)))
print('worst design relative errors', {'shift': float(shift_error), 'sensor': float(sensor_error)})
assert max(shift_error, sensor_error) < 1e-10

for wing in [0, 1]:
    st = copy.deepcopy(s)
    c = st['sensor_calibration_sweeps']
    gain, offset = np.array([.78,1.23,.67,1.46]), np.array([.21,-.17,.35,-.26])
    cb, rb = c['bank_id'], st['row_bank_id']
    c['raw_readout'] = gain[cb]*c['ref_signal'] + offset[cb]
    st['y_obs_raw'] = gain[rb]*s['y_obs_raw']+offset[rb]
    keep = np.ones(cb.size, dtype=bool)
    for bank in range(4):
        ix = np.flatnonzero(cb == bank)
        if bank//2 == wing:
            extreme = ix[np.argsort(c['ref_signal'][ix])[[0,-1]]]
            c['raw_readout'][extreme] += np.array([4., -5.])
        else:
            keep[ix[2:]] = False
    for key in c:
        c[key] = c[key][keep]
    print('EMI wing / sparse other wing', wing, json.dumps(check(st)))

rng = np.random.default_rng(831)
worst_gap = worst_obj = worst_feas = 0.
for z in range(144):
    n = 12
    A = rng.normal(size=(24,n))*10.**rng.uniform(-1,1)
    if z%3 == 0:
        A[:,6:] = A[:,:6]
    y = rng.normal(size=24)*10.**rng.uniform(-1,2)
    r, u = .04, 1.
    b = [0., n*u, 1e-8, n*u-1e-8, 1., 6., 11., 2.6, 9.45][z%9]
    w, nu = mod._box_simplex(A,y,r,u,b)
    def obj(w):
        e = A@w-y
        return .5*(e@e+r*(w@w))
    def grad(w): return A.T@(A@w-y)+r*w
    g = grad(w)
    gap = (np.maximum(g-nu,0.)@w + np.maximum(nu-g,0.)@(u-w)) - nu*(b-w.sum())
    feas = max(abs(w.sum()-b), -w.min(), w.max()-u)
    worst_gap, worst_feas = max(worst_gap, abs(gap)), max(worst_feas, feas)
    if z%9 in [4,5,6,7,8] and z < 45:
        alt = minimize(obj, np.full(n,b/n), jac=grad, method='SLSQP', bounds=[(0.,u)]*n, constraints=[{'type':'eq','fun':lambda w: w.sum()-b,'jac':lambda w: np.ones(n)}], options={'ftol':1e-11,'maxiter':500})
        if alt.success:
            worst_obj = max(worst_obj, obj(w)-alt.fun)
assert worst_gap < 1e-7 and worst_feas < 1e-9 and worst_obj < 1e-7
print('QP stress', {'cases': 144, 'max_gap': float(worst_gap), 'max_feasibility_error': float(worst_feas), 'max_objective_excess_vs_SLSQP': float(worst_obj)})
