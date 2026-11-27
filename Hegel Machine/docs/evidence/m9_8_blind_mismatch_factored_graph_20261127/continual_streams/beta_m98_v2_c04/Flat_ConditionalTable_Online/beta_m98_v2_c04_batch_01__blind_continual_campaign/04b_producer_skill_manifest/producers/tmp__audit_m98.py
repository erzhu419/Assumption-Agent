import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)

for item in items:
    s = item['study']
    nx, ny = s['nx'], s['ny']
    n = nx * ny
    shifts = s['mu_shifts']
    rhs = np.column_stack([v.ravel(order='F') for v in s['source_modes']])
    A = np.diag(s['V_reaction_2d'].ravel(order='F'))
    for axis, size, stride in [('x', nx, 1), ('y', ny, nx)]:
        h = s['L'+axis] / (size-1)
        k, v = s['kappa_'+axis], s['vel_'+axis]
        for p in range(n):
            j = (p//stride) % size
            if j in (0, size-1):
                A[p,p] += k[j]/h**2
                A[p,p+(stride if j == 0 else -stride)] -= k[j]/h**2
            else:
                left = (k[j]+k[j-1])/(2*h**2) + max(v[j],0)/h
                right = (k[j]+k[j+1])/(2*h**2) + max(-v[j],0)/h
                A[p,p-stride] -= left
                A[p,p+stride] -= right
                A[p,p] += left+right
    ix, iy = s['sensor_ij'].T
    fields = [np.linalg.solve(A + shift*np.eye(n), rhs) for shift in shifts]
    S = np.vstack([field[ix+nx*iy] for field in fields])
    candidate_S, _ = m._forward(s)
    np.testing.assert_allclose(candidate_S, S, atol=1e-12, rtol=1e-10)
    r = s['sensor_calibration_sweeps']['ref_rosette']
    raw = s['sensor_calibration_sweeps']['raw_rosette']
    affine = np.column_stack((np.ones(len(r)), r))
    terms = [r[:,i]*r[:,j] for i in range(4) for j in range(i,4)]
    quadratic = np.column_stack((affine, *terms))
    groups = np.random.default_rng(2401).permutation(len(r)) % 4
    for name, design in [('affine', affine), ('quadratic', quadratic)]:
        rms = []
        for k in range(4):
            test = groups == k
            co = np.linalg.lstsq(design[~test], raw[~test], rcond=None)[0]
            rms.append(np.sqrt(np.mean((design[test]@co-raw[test])**2,axis=0)))
        print(name, 'worst held-out sweep-channel RMS', np.max(rms))
    banks = np.asarray(s['row_bank_id']).reshape(len(shifts),-1)
    worst = 0.
    for p in item.get('campaign_diagnostics') or []:
        y = m._calibrate(dict(s,y_obs_raw=p['y_pulse_raw']))
        e = (S @ p['w_pulse'] - y).reshape(banks.shape)
        for errors, labels in zip(e, banks):
            worst = max(worst, *(np.sqrt(np.mean(errors[labels==b]**2)) for b in np.unique(labels)))
    assert worst < 1e-5, worst
    print('worst pulse/shift/bank RMS', worst)
    y = m._calibrate(s)
    Z = s['Z_nuisance']
    v = np.column_stack((S, y))
    pv = v - Z @ np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T@v)
    X, y = pv[:,:-1], pv[:,-1]
    ridge, cap, budget = (s[k] for k in ('source_ridge','u_box','total_budget'))
    H, q = X.T@X + ridge*np.eye(X.shape[1]), -X.T@y
    def objective(w):
        e = X@w-y
        return .5*(e@e + ridge*(w@w))
    best = np.inf
    for status in itertools.product((-1,0,1), repeat=X.shape[1]):
        status = np.array(status)
        free, upper = status==0, status==1
        rest = budget - cap*upper.sum()
        count = free.sum()
        if rest < -1e-12 or rest > count*cap+1e-12:
            continue
        w = cap*upper.astype(float)
        if count:
            one = np.ones((count,1))
            K = np.block([[H[np.ix_(free,free)], -one], [one.T, np.zeros((1,1))]])
            w[free] = np.linalg.solve(K, np.r_[-q[free]-H[free]@w, rest])[:-1]
        if w.min() >= -1e-12 and w.max() <= cap+1e-12:
            best = min(best, objective(w))
    out = m.solve_batch(s)
    w, c = out['weights'], out['kkt_certificate']
    lo, hi = np.asarray(c['lam_lower']), np.asarray(c['lam_upper'])
    g = H@w+q
    err = max(np.max(np.abs(g-c['nu_sum']-lo+hi)), np.max(np.abs(w*lo)), np.max(np.abs((cap-w)*hi)), abs(w.sum()-budget), max(0,-w.min(),w.max()-cap,-lo.min(),-hi.min()))
    assert err < 1e-10, err
    np.testing.assert_allclose(c['weights'], w, atol=1e-12, rtol=0)
    np.testing.assert_allclose(out['terminal_field'], (fields[-1]@w).reshape((nx,ny),order='F'), atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose([objective(w), c['primal_objective'], c['dual_lower_bound']], best, atol=1e-10, rtol=0)
    print('independent forward max difference', np.max(np.abs(candidate_S-S)), 'KKT max error', err, 'objective error vs face enumeration', abs(objective(w)-best))
