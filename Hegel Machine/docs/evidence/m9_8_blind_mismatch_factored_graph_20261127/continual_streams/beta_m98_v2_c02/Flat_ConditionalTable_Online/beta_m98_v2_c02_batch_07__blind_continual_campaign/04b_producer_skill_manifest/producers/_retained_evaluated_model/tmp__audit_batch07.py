import importlib.util
import pickle
import numpy as np
from scipy.optimize import root
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
S, _ = m._forward(s)
x = s['sensor_calibration_sweeps']['ref_rosette']
y = s['sensor_calibration_sweeps']['raw_rosette']
D = m._rosette_features(x)
# The observed sparse structure has only self cubic, cyclic-neighbor affine,
# and a product with the next channel. All coefficients are fitted in each fold.
cols = [[0, 1+j, 1+(j+1)%4, 1+(j-1)%4, 5+j, 9+j] for j in range(4)]
Dplus = np.column_stack((D, x*x, x[:,0]*x[:,2], x[:,1]*x[:,3]))
truth = [S @ p['w_pulse'] for p in item['campaign_diagnostics']]
for model in ('current', 'channel_sparse', 'extra_quadratic'):
    held = []
    pulse = []
    for test in np.array_split(np.random.default_rng(42).permutation(len(x)), 5):
        train = np.setdiff1d(np.arange(len(x)), test)
        # Detect contamination without looking at the held-out rows.
        e = abs(D[train] @ np.linalg.lstsq(D[train], y[train], rcond=None)[0]-y[train]).max(1)
        keep = train[e <= 5*np.median(e)]
        E = Dplus if model == 'extra_quadratic' else D
        c = np.zeros((E.shape[1], 4))
        for j in range(4):
            ix = cols[j] if model == 'channel_sparse' else np.arange(E.shape[1])
            c[ix,j] = np.linalg.lstsq(E[keep][:,ix], y[keep,j], rcond=None)[0]
        held.append(E[test] @ c - y[test])
        def f(z):
            a = m._rosette_features(z[None,:])[0]
            if model == 'extra_quadratic':
                a = np.r_[a, z*z, z[0]*z[2], z[1]*z[3]]
            return a @ c
        for p, target in zip(item['campaign_diagnostics'],truth):
            raw = p['y_pulse_raw'].reshape(-1,4)
            guesses = np.linalg.solve(c[1:5].T, (raw-c[0]).T).T
            inv = [root(lambda z: f(z)-r, g, tol=1e-10) for r,g in zip(raw,guesses)]
            assert max(np.max(abs(v.fun)) for v in inv) < 1e-9
            err = (np.array([v.x for v in inv]).ravel()-target).reshape(len(s['mu_shifts']),-1,4)
            pulse.append(np.sqrt(np.mean(err**2,axis=1)))
    err = abs(np.vstack(held))
    print(model,'worst bank OOF p95',np.quantile(err,.95,axis=0).max(),'worst pulse x shift x bank RMSE',np.max(pulse))
Z = s['Z_nuisance']
project = lambda a: a-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@a)
# Restrict sensitivity to perturbations that preserve the equality budget.
B = np.eye(S.shape[1])[:,:-1]; B[-1] = -1
before = np.linalg.svd(S@B,compute_uv=False)
after = np.linalg.svd(project(S)@B,compute_uv=False)
print('projected and unprojected source tangent singular values:',after,before)
print('pulse groups:',len(truth),'shift groups:',len(s['mu_shifts']),'nuisance columns:',Z.shape[1])
