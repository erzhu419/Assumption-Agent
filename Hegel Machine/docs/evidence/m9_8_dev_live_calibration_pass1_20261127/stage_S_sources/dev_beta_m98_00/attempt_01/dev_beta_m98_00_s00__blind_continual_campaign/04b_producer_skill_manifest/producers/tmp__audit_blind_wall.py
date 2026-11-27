import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    studies = pickle.load(f)
for item in studies:
    s = item['study']
    source = s['source_modes']
    cal = m._calibration(s['sensor_calibration_sweeps'])
    y = cal(s['y_obs_raw'])
    z = s['Z_nuisance']
    nr, rr, cap, budget = (s[k] for k in ('nuisance_ridge','source_ridge','u_box','total_budget'))
    pts = np.asarray(s['sensor_ij'])
    if len(pts) != len(y):
        pts = np.tile(pts, (len(s['mu_shifts']), 1))
    _, loc = np.unique(pts, axis=0, return_inverse=True)
    folds = loc % 4
    pulse = s.get('reference_tracer_pulses') or []
    nominal = m._response(s, np.ones(8), source)[0]
    y_error = {tag: [] for tag in ('nominal','blind')}
    pulse_error = {tag: [] for tag in y_error}
    for k in np.unique(folds):
        train = folds != k
        zt = z[train]
        p = np.linalg.solve(zt.T @ zt + nr*np.eye(z.shape[1]), zt.T)
        yt = y[train]
        yp = yt - zt @ (p @ yt)
        def fit(S):
            St = S[train]
            X = St - zt @ (p @ St)
            w = m._weights(X, yp, rr, cap, budget)
            gamma = p @ (yt - St @ w)
            return w, gamma, X @ w - yp
        def response(t):
            wall = np.ones(8)
            wall[[0,1]], wall[[4,5]] = t
            return m._response(s, wall, source)[0]
        def residual(t):
            w, _, r = fit(response(t))
            return np.r_[r, np.sqrt(rr)*w]
        opt = least_squares(residual, [1.,1.], bounds=(0.25, 4.), max_nfev=30)
        for tag, S in [('nominal',nominal),('blind',response(opt.x))]:
            w, gamma, _ = fit(S)
            e = (S @ w + z @ gamma - y)[~train]
            y_error[tag].append(np.sqrt(np.mean(e**2)))
            if pulse:
                pulse_error[tag].append(max(np.sqrt(np.mean(((S @ q['w_pulse'] - cal(q['y_pulse_raw']))[~train])**2)) for q in pulse))
        print('fold', k, 'blind Robin scale factors', np.round(opt.x, 5), 'held-out measurement RMSE nominal/blind', *(y_error[tag][-1] for tag in y_error))
    print('Worst spatial fold RMSE, measurements:', {tag: max(v) for tag,v in y_error.items()})
    print('Worst held-out spatial fold and independent pulse RMSE:', {tag: max(v) for tag,v in pulse_error.items() if v})
