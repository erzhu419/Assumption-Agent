import importlib.util, pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
a = Path('/tmp/hegel_continual_archive')
mods = {}
for name, suffix in [('robin', 's00'), ('reflecting', 'batch_01')]:
    spec = importlib.util.spec_from_file_location(name, a / f'dev_beta_m98_00_{suffix}_solver.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mods[name] = mod

c = s['sensor_calibration_sweeps']
r, raw = c['ref_rosette'], c['raw_rosette']
print('Calibration range:', r.min(0), r.max(0), 'obs raw range:', s['y_obs_raw'].min(), s['y_obs_raw'].max())
def basis(r, degree):
    cols = [np.ones(len(r)), *r.T]
    if degree >= 2:
        cols += [r[:,i]*r[:,j] for i in range(4) for j in range(i,4)]
    if degree >= 3:
        cols += [r[:,i]*r[:,j]*r[:,k] for i in range(4) for j in range(i,4) for k in range(j,4)]
    return np.column_stack(cols)
for degree in (1,2,3):
    x = basis(r, degree)
    e = np.zeros_like(raw)
    for tr, te in KFold(5, shuffle=True, random_state=17).split(r):
        e[te] = raw[te] - x[te] @ np.linalg.lstsq(x[tr], raw[tr], rcond=None)[0]
    print('cal degree', degree, 'OOF channel RMSE', np.sqrt(np.mean(e**2, axis=0)), 'max', np.max(np.abs(e)))
print('Affine coefficients:', np.linalg.lstsq(basis(r,1), raw, rcond=None)[0])
for k,v in c['single_bank_sweeps'].items():
    print('bank sweep', k, v.shape, 'range', np.min(v), np.max(v))
print('row banks first 16:', s['row_bank_id'][:16])

z = s['Z_nuisance']; n = len(z)
y = mods['robin']._calibrate(s)
print('archived calibration difference', np.max(np.abs(y-mods['reflecting']._calibrate(s))))
ridge = s['source_ridge']; cap = s['u_box']; budget = s['total_budget']; nr = s['nuisance_ridge']
ns = len(s['sensor_ij']); shift = np.arange(n)//ns; group = np.arange(n)%ns//4
masks = [('rosette', group==i) for i in np.unique(group)] + [('shift', shift==i) for i in np.unique(shift)]
for name, mod in mods.items():
    D, U = mod._design(s)
    def fit(keep):
        zz, dd, yy = z[keep], D[keep], y[keep]
        p = np.linalg.solve(zz.T@zz + nr*np.eye(z.shape[1]), zz.T)
        d = dd - zz@(p@dd); t = yy - zz@(p@yy)
        w, nu = mod._box_qp(d.T@d + ridge*np.eye(D.shape[1]), -d.T@t, cap,budget)
        return w, p@(yy-dd@w)
    w,g = fit(np.ones(n,bool)); e = D@w+z@g-y
    print(name,'full RMSE',np.sqrt(np.mean(e**2)), 'weights',w.round(6), 'terminal norm', np.linalg.norm(np.tensordot(w,U,axes=1)))
    for kind in ('rosette','shift'):
        err = []
        for label, test in masks:
            if label != kind: continue
            w,g = fit(~test)
            e = D[test]@w+z[test]@g-y[test]
            err.append(np.sqrt(np.mean(e**2)))
        print(name,kind,'RMSE by held-out slice',np.array(err).round(7),'pooled',np.sqrt(np.mean(np.square(err))))
