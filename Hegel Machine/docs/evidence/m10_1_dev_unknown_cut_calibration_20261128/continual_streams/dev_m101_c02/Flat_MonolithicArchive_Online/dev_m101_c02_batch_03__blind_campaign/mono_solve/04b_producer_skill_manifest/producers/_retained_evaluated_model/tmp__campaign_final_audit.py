import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
tr = s['train']
print('Study fields:', sorted(s), 'train fields:', sorted(tr), 'telemetry shape:', np.shape(s['telemetry_vec']))
z = m._coordinates(s, tr['x'])
y = tr['y']
d = m._scalar_design(z)
p = d[:, -1]
pdiff = z[:, 0]*z[:, 2] - z[:, 1]*z[:, 3]
# A pairwise sum or difference squared produces the corresponding product and diagonal terms.
extra = np.column_stack((pdiff, z**2))
a = s['panel_a']
keep = np.max(np.abs(a['v']-a['u']), axis=1)<3
u, v = a['u'][keep], a['v'][keep]
f = m._calibration_design(u)
blocks = np.stack((u, np.roll(u, -1, axis=1), u-np.roll(u, -1, axis=1)), axis=-1).reshape(-1, 3)**2
actual = v.ravel()
rows = []
for k in range(len(u)):
    i = np.repeat(np.arange(len(u)) != k, 4)
    c = m._ls(f[i], actual[i])
    base = f[~i] @ c
    deviations = []
    for q2 in blocks.T:
        # Profile the numerator on training panel rows only.
        def fit(val):
            g = f/(1+val*q2[:, None])
            return g, m._ls(g[i], actual[i])
        def res(val):
            g, coef = fit(val[0]); return g[i]@coef-actual[i]
        val = least_squares(res, [0.001], bounds=(0, 5)).x[0]
        g, coef = fit(val)
        deviations.append(g[~i]@coef)
    rows.append(np.array([base, *deviations])-actual[~i])
print('Panel A leave-one-out RMSE [cyclic, rational self, rational neighbor, rational difference]:', np.sqrt(np.mean(np.square(rows), axis=(0,2))))
ids = np.arange(len(z))
order = np.argsort(np.linalg.norm(z, axis=1))
folds = [('56/16', ids[:56], ids[56:]), ('large radius', order[:-12], order[-12:])]
folds += [(f'random {k}', np.setdiff1d(ids, j), j) for k,j in enumerate(np.array_split(np.random.default_rng(14).permutation(ids), 5))]
errors = []
for name, i, j in folds:
    # Profile three possible denominator directions on each training fold and select with inner CV only.
    qs = [z[:,0]-z[:,2], z[:,1]-z[:,3], z[:,0]+z[:,2], z[:,1]+z[:,3], np.linalg.norm(z, axis=1)]
    def fit_rational(q, ii, jj):
        def model(k):
            g = d/(1+k*q[:,None]**2)
            return g, m._ls(g[ii], y[ii])
        def resid(k):
            g, c = model(k[0]); return g[ii]@c-y[ii]
        g, c = model(least_squares(resid, [0.001], bounds=(0, 5)).x[0])
        return g[jj] @ c
    inner = np.array_split(np.random.default_rng(35).permutation(i), 3)
    q = min(qs, key=lambda q: np.mean([np.mean(abs(fit_rational(q,np.setdiff1d(i,b),b)-y[b])) for b in inner]))
    f2 = np.column_stack((d, extra))
    preds = [d[j]@m._ls(d[i], y[i]), f2[j]@m._ls(f2[i], y[i]), fit_rational(q,i,j)]
    fold_err = np.mean(abs(np.array(preds)-y[j]), axis=1)
    errors.append(fold_err)
    print(name, 'scalar MAE [paired sum, expanded pair squares, rational]:', np.round(fold_err, 7))
print('Worst scalar MAE:', np.max(errors, axis=0))
