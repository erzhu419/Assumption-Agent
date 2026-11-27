import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
cal = s['sensor_calibration_sweeps']
x, y = cal['ref_rosette'], cal['raw_rosette']
D = m._rosette_features(x)
S, _ = m._forward(s)
pulses = [(np.asarray(S @ p['w_pulse']).reshape(-1,4), p['y_pulse_raw'].reshape(-1,4)) for p in item['campaign_diagnostics']]

def q2(z, kind):
    if kind == 'self':
        return z*z
    sign = 1 if kind == 'neighbor_sum' else -1
    return (z + sign*np.roll(z,-1,axis=1))**2

folds = np.array_split(np.random.default_rng(42).permutation(len(x)),5)
for kind in ('polynomial','self','neighbor_sum','neighbor_difference'):
    held, slices, denominator_min = [], [], np.inf
    for test in folds:
        train = np.setdiff1d(np.arange(len(x)),test)
        c = np.linalg.lstsq(D[train],y[train],rcond=None)[0]
        e = np.max(abs(D[train]@c-y[train]),axis=1)
        keep = train[e <= 5*np.median(e)]
        a = np.zeros(4)
        if kind == 'polynomial':
            c = np.linalg.lstsq(D[keep],y[keep],rcond=None)[0]
        else:
            q = q2(x[keep],kind)
            for j in range(4):
                fit = np.linalg.lstsq(np.column_stack((D[keep],-y[keep,j]*q[:,j])),y[keep,j],rcond=None)[0]
                c[:,j], a[j] = fit[:-1], fit[-1]
        def predict(z):
            global denominator_min
            den = np.ones_like(z) if kind == 'polynomial' else 1+q2(z,kind)*a
            denominator_min = min(denominator_min,den.min())
            return (m._rosette_features(z)@c)/den
        held.append(abs(predict(x[test])-y[test]))
        for z, raw in pulses:
            residual = (predict(z)-raw).reshape(len(s['mu_shifts']),-1,4)
            slices.append(np.max(np.sqrt(np.mean(residual**2,axis=1))))
    print(kind, 'worst bank sweep p95:', np.quantile(np.vstack(held),.95,axis=0).max(), 'worst raw pulse/shift/bank RMSE:',max(slices),'min tested denominator:',denominator_min)
