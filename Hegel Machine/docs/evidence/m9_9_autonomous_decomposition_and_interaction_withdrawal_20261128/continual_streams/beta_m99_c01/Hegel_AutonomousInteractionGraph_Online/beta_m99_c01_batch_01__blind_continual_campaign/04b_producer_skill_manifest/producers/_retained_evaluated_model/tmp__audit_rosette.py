import importlib.util
import itertools
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'batch_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
for item in pickle.load(open(root + 'public_batch.pkl', 'rb')):
    d = item['study']
    cal = d['sensor_calibration_sweeps']
    x, y = cal['ref_rosette'], cal['raw_rosette']
    single = cal['single_bank_sweeps']
    b, sx, sy = (np.asarray(single[k]) for k in ['bank_id', 'ref_signal', 'raw_readout'])
    S, _ = solver._forward(d)
    z = d['Z_nuisance']
    P = np.eye(len(z)) - z @ np.linalg.solve(z.T @ z + d['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    pairs = list(itertools.combinations(range(4), 2))
    triples = list(itertools.combinations(range(4), 3))
    def design(a, family):
        a = np.atleast_2d(a)
        cols = [np.ones(len(a)), *a.T]
        if family == 'interactions':
            cols += [*(a**2).T, *[a[:, i]*a[:, j] for i,j in pairs], *[a[:, i]*a[:,j]*a[:,k] for i,j,k in triples]]
        if family == 'rational':
            cols += [a[:, i]/(1+(a[:,j]+sign*a[:,k])**2) for i in range(4) for j,k in pairs for sign in (-1,1)]
        return np.column_stack(cols)
    def fit(train, family):
        if family in ('diagonal', 'pooled'):
            coeff = []
            for i in range(4):
                xx, yy = x[train, i], y[train, i]
                if family == 'pooled':
                    xx, yy = np.r_[xx, sx[b==i]], np.r_[yy, sy[b==i]]
                coeff.append(np.linalg.lstsq(np.column_stack([np.ones(len(xx)),xx]),yy,rcond=None)[0])
            coeff = np.asarray(coeff)
            return lambda a: coeff[:,0] + a*coeff[:,1]
        D = design(x[train], family)
        if D.shape[1] >= D.shape[0] or np.linalg.matrix_rank(D) < D.shape[1]:
            return None
        c = np.linalg.lstsq(D,y[train],rcond=None)[0]
        return lambda a: design(a, family) @ c
    for name in ['diagonal','pooled','affine','interactions','rational']:
        errs = []
        for fold in range(5):
            test = np.arange(len(x))%5==fold
            model = fit(~test,name)
            if model is not None:
                errs.append(np.sqrt(np.mean((model(x[test])-y[test])**2,axis=0)).max())
        if len(errs)!=5:
            print(name,'not identifiable across folds'); continue
        model = fit(np.ones(len(x),dtype=bool),name)
        structural, projected, means = [], [], []
        for pulse in item.get('campaign_diagnostics') or []:
            expected = (S @ pulse['w_pulse']).reshape(-1,4)
            raw = pulse['y_pulse_raw'].reshape(-1,4)
            raw_error = (model(expected)-raw).reshape(len(d['mu_shifts']),-1,4)
            structural.extend(np.sqrt(np.mean(raw_error**2,axis=1)).ravel())
            means.extend(raw_error.mean(axis=1).ravel())
            inverse = np.array([least_squares(lambda q: (model(q)-r).ravel(),p,gtol=1e-13).x for p,r in zip(expected,raw)])
            e = (inverse-expected).ravel()
            projected.append(np.sqrt(np.mean((P@e)**2)))
        print(name, 'worst held-out raw sweep fold/bank',max(errs),'worst raw pulse shift/bank',max(structural,default=0),'worst projected pulse RMSE',max(projected,default=0),'max raw pulse shift/bank bias',max(np.abs(means),default=0))
    base = fit(np.ones(len(x),dtype=bool),'diagonal')
    single_err = base(np.eye(4)[b] * sx[:,None])[np.arange(len(b)),b] - sy
    print('independent single-bank RMSE', [float(np.sqrt(np.mean(single_err[b==i]**2))) for i in range(4)])
