import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
c = d['calibration_load_cells']
a = c['bridge_raw']
b = c['true_channel_loads'].sum(axis=1)
tr, te = d['train'], d['test']
cal = m._stress_calibration(c)
loads = np.concatenate((cal(te['past_channel_raw']).ravel(), cal(te['hold_channel_raw'])))
print('Calibrated total stress range; deployment range:', np.round([b.min(), b.max(), loads.min(), loads.max()], 5))

def compare(outer):
    scores = []
    for held in outer:
        kept = np.setdiff1d(np.arange(len(a)), held)
        center, scale = a[kept].mean(axis=0), a[kept].std(axis=0)
        x = (a-center)/scale
        base = np.c_[np.ones(len(x)), x]
        options = [base]
        for axis in [*x.T, *(x[:,i]+sign*x[:,j] for i in range(4) for j in range(i+1,4) for sign in (-1,1))]:
            for strength in (.001, .01, .1):
                options.append(np.c_[base, x/(1+strength*axis[:,None]**2)])
        def mse(f, train, test):
            coef = np.linalg.lstsq(f[train], b[train], rcond=None)[0]
            return np.mean((f[test]@coef-b[test])**2)
        inner = np.array_split(kept, 4)
        idx = np.argmin([np.mean([mse(f,np.setdiff1d(kept,test),test) for test in inner]) for f in options])
        scores.append((np.sqrt(mse(base, kept, held)),np.sqrt(mse(options[idx], kept, held))))
    return np.round(scores,7)
order = np.argsort(b)
n = max(2,len(b)//5)
print('Extreme stress fold RMSE (affine, nested rational challenger):', compare((order[:n],order[-n:])).tolist())
x, _ = m._standardize(tr, te)
y = tr['strain_curves_obs']-tr['strain_0'][:,None]
r = tr['instant_rate_obs']
mat = m._design(tr,x,cal,d['past_dt'],d['eval_times'])
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
e, q = np.zeros_like(y), np.zeros_like(r)
for lab in np.unique(tr['lab_id']):
    h = tr['lab_id']==lab
    tau, theta = m._fit(lambda t: tuple(z[~h] for z in mat(t)), y[~h], r[~h])
    strain, rate = mat(tau)
    e[h],q[h] = strain[h]@theta-y[h],rate[h]@theta-r[h]
for label, h in [('observed',~missing),('imputed',missing)]:
    if np.any(h):
        print(label, h.sum(), 'OOF strain/rate MedianAE', np.round([np.median(abs(e[h])),np.median(abs(q[h]))],5))
