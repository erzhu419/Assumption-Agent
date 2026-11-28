import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
b = c['single_bank_sweeps']
# The measured +14.41 isolated readout is outside the clean sweep range by >12.
# Exclude it from the scoring population but leave it in every applicable training fold.
A, Y, folds, clean = [], [], [], []
for j in range(4):
    idx = b['bank_id'] == j
    bx = np.zeros((idx.sum(), 4))
    bx[:, j] = b['ref_signal'][idx]
    A.append(m._design(np.vstack((x, bx)), j))
    Y.append(np.r_[y[:, j], b['raw_readout'][idx]])
    folds.append(np.r_[np.arange(len(x)) % 5, np.arange(len(bx)) % 5])
    clean.append(np.abs(Y[-1]) < 5)
A, Y, folds, clean = map(np.array, (A, Y, folds, clean))
for j, col in enumerate(Y):
    bad = np.flatnonzero(~clean[j]); print('bank', j, 'gross reads', list(zip(bad.tolist(), col[bad].round(4).tolist())))
print('largest retained raw magnitude by bank', np.max(np.where(clean, abs(Y), 0), axis=1))

def fit(keep, model):
    coef = []
    inlier = []
    for D, q, train in zip(A, Y, keep):
        Dt, qt = D[train], q[train]
        f = np.linalg.lstsq(Dt, qt, rcond=None)[0]
        for _ in range(6):
            e = qt - Dt @ f
            scale = max(1e-8, 1.4826 * np.median(abs(e - np.median(e))))
            w = 1 / np.hypot(1, e / (2.5 * scale))
            f = np.linalg.lstsq(Dt*w[:, None], qt*w, rcond=None)[0]
        e = qt - Dt @ f
        good = train.copy()
        good[train] = abs(e) < max(2e-4, 8 * 1.4826 * np.median(abs(e - np.median(e))))
        inlier.append(good)
        coef.append(np.linalg.lstsq(D[good], q[good], rcond=None)[0])
    coef = np.array(coef)
    if model == 'independent':
        return np.einsum('jnp,jp->jn', A, coef)
    def predict(t):
        z = A[:,:,1] + A[:,:,2:] @ t[8:12]
        if model == 'shared_rational':
            z = z / (1 + t[12]*A[:,:,1]**2)
        return t[:4,None] + t[4:8,None]*z
    start = np.r_[coef[:,0], coef[:,1], np.mean(coef[:,2:] / coef[:,1,None], axis=0)]
    if model == 'shared_rational':
        start = np.r_[start, 0.]
    mask = np.array(inlier)
    params = least_squares(lambda t: (predict(t)-Y)[mask], start).x
    if model == 'shared_rational':
        print('rational coefficient', params[-1])
    return predict(params)

for model in ('independent', 'shared', 'shared_rational'):
    pred = np.empty_like(Y)
    for fold in range(5):
        keep = folds != fold
        pred[~keep] = fit(keep, model)[~keep]
    err = pred-Y
    print(model, 'gross absolute residuals', np.round(abs(err[~clean]), 4).tolist())
    print(model, 'clean bank RMS', [float(np.sqrt(np.mean(v[k]**2))) for v,k in zip(err,clean)], 'worst clean absolute', np.max(abs(err[clean])))
    for name, sl in [('rosette',slice(None,len(x))),('single',slice(len(x),None))]:
        z, k = err[:,sl], clean[:,sl]
        print(name, 'worst bank RMS', max(np.sqrt(np.sum(np.where(k,z*z,0),axis=1) / k.sum(axis=1))))
