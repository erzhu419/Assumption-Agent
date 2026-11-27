import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    s = pickle.load(f)[0]['study']
a = s['sensor_calibration_sweeps']
x, y = a['ref_rosette'], a['raw_rosette']
basis = m._rosette_basis(x)
# Test the observed common gain-normalized cyclic law, and whether a rational
# interaction in the difference coordinate explains any residual structure.
def predict(p, d, rational):
    v = d[..., 1] + np.einsum('njk,k->nj', d[..., 2:], p[8:12])
    if rational:
        v /= 1 + p[12] * (d[..., 1] - d[..., 2])**2
    return p[:4] + v*p[4:8]

def fit(train, rational):
    c = m._fit_rosette(x[train], y[train])
    r = np.linalg.norm(y[train] - np.einsum('njk,jk->nj', basis[train], c), axis=1)
    clean = r <= max(8*np.median(r), 1e-10)
    p0 = np.r_[c[:,0], c[:,1], np.mean(c[:,2:]/c[:,1,None], axis=0)]
    if rational:
        p0 = np.r_[p0, 0.]
    p = least_squares(lambda p: (predict(p, basis[train][clean], rational)-y[train][clean]).ravel(), p0, xtol=1e-13, ftol=1e-13, gtol=1e-13).x
    return c, p, clean

folds = np.array_split(np.arange(len(x)),4) + [np.flatnonzero(np.count_nonzero(x,axis=1)==1)]
errors = [[], [], []]
ratios = []
for test in folds:
    train = ~np.isin(np.arange(len(x)), test)
    c, p, _ = fit(train, False)
    _, q, _ = fit(train, True)
    forecasts = [np.einsum('njk,jk->nj', basis[test], c), predict(p,basis[test],False), predict(q,basis[test],True)]
    for group, forecast in zip(errors, forecasts):
        e = np.max(abs(y[test]-forecast), axis=1)
        group.append((float(e[e<1].max()), int(np.sum(e>=1))))
    ratios.append(q[-1])
for name, result in zip(('cyclic', 'shared gain-normalized', 'shared rational difference'), errors):
    print(name, 'blocked (max clean, gross count)', result, 'worst clean', max(v[0] for v in result))
print('rational denominator coefficients by fold', ratios)
b = a['single_bank_sweeps']; j, u = b['bank_id'], b['ref_signal']
for rational in (False, True):
    c, p, _ = fit(np.ones(len(x), dtype=bool), rational)
    refs = np.zeros((len(u),4)); refs[np.arange(len(u)),j] = u
    e = abs(predict(p,m._rosette_basis(refs),rational)[np.arange(len(u)),j]-b['raw_readout'])
    print('independent bank, rational=',rational,'worst clean',e[e<1].max(),'gross count',np.sum(e>=1))

S, _ = m._forward(s)
ycal = m._calibrate(s); Z = s['Z_nuisance']; w = m.solve_batch(s)['weights']
gamma = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@(ycal-S@w))
res = (ycal-S@w-Z@gamma).reshape(len(s['mu_shifts']),-1,4)
print('calibrated signed residual means by shift x bank',res.mean(axis=1).round(6).tolist())
print('calibrated residual RMSE by shift x bank',np.sqrt((res**2).mean(axis=1)).round(6).tolist())
