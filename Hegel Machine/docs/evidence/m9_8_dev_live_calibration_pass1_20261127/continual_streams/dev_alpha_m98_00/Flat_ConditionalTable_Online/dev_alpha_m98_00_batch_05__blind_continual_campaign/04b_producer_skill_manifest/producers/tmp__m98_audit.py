import importlib.util, pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['dataset']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
cal = m._calibration(d['calibration_load_cells'])
x, _ = m._features(tr, te)
a = m._inputs(tr, x, cal)
lab, st = tr['lab_id'], tr['stratum']
obs = np.c_[tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']]
z = tr['aux_chamber_telemetry']
# Center within labs without using any response data. Compare a fixed telemetry
# basis to the null using outer labs only, scoring each lab and stratum separately.
zc = z.copy()
for l in np.unique(lab):
    zc[lab == l] -= np.mean(z[lab == l], axis=0)
rows = []
for l in np.unique(lab):
    q = lab == l
    f = ~q
    tau, th, scale = m._material({k: v[f] for k,v in tr.items()}, tuple(v[f] for v in a), dt, t)
    y, r = m._predict(a, dt, t, tau, th)
    e = obs - np.c_[y, r]
    # Compare a telemetry-driven drift with precisely the required derivative.
    g = np.r_[t, 1.]
    s = np.broadcast_to(np.asarray(scale).ravel()[:2], (2,))
    w = 1/np.r_[np.repeat(s[0], len(t)), s[1]]**2
    v = (np.clip(e[f], -3/np.sqrt(w), 3/np.sqrt(w)) @ (g*w)) / ((g*g)@w)
    zf = zc[f]
    beta = np.linalg.solve(zf.T@zf + 5*np.eye(zf.shape[1]), zf.T@v)
    ea = e - (zc@beta)[:, None]*g
    def stats(j, residual):
        return [np.median(np.abs(residual[j,:-1])),np.median(np.abs(residual[j,-1]))]
    print('lab', l, 'baseline', stats(q,e), 'centered telemetry', stats(q,ea))
    for h in np.unique(st[q]):
        j = q & (st == h)
        rows.append([*stats(j,e), *stats(j,ea)])
print('Worst lab-stratum median abs strain/rate, baseline then telemetry:', np.max(rows, axis=0))

# Check pairwise difference and low-order rational forward bridge alternatives
# against direct held-out total load, not just the calibration training loss.
u, raw = (d['calibration_load_cells'][k] for k in ('true_channel_loads','bridge_raw'))
folds = np.array_split(np.random.default_rng(14).permutation(len(u)), 5)
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
res = []
for q in folds:
    f = np.setdiff1d(np.arange(len(u)), q)
    z = u / u[f].std(axis=0)
    A = np.c_[np.ones(len(u)), z]
    delta = np.column_stack([z[:, i] - z[:, j] for i,j in pairs])
    # Linear differences are aliases; their squares add the complete pairwise quadratic basis.
    assert np.max(abs(delta - A @ np.linalg.lstsq(A[f],delta[f],rcond=None)[0])) < 1e-10
    B = np.c_[A, z*z, delta*delta, z/(1+z*z)]
    # Diagnose a forward-model failure at fixed true load before adding inversion complexity.
    v = []
    for design in (A,B):
        C = np.linalg.lstsq(design[f],raw[f],rcond=None)[0]
        e = raw[q]-design[q]@C
        # Convert forward residual to a first-order total-load error using the common affine map.
        J = np.linalg.lstsq(A[f],raw[f],rcond=None)[0][1:]
        v.append(np.max(abs(e @ np.linalg.solve(J,u[f].std(axis=0)))))
    res.append(v)
print('Worst held-out forward residual in total-load units, affine / expanded:', np.max(res,axis=0))
