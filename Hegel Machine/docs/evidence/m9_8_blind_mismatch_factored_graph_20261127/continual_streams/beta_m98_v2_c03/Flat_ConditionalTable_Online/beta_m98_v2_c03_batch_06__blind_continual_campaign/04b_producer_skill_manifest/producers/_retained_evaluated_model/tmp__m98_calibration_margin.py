import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('batch_solver', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def features(x, i, model):
    base = m._features(x, i)
    own, neighbor = x[..., i], x[..., (i+1)%4]
    if model == 'archived':
        return base
    if model == 'pairwise':
        return np.column_stack((base, own**2, neighbor**2, (own-neighbor)**3, (own+neighbor)**3))
    if model == 'three_way':
        return np.column_stack((base, own*neighbor*x[..., (i+2)%4]))
    return np.column_stack((base, own/(1+0.25*(own-neighbor)**2), own/(1+0.25*(own+neighbor)**2)))

for item in pickle.load(open(root+'public_batch.pkl','rb')):
    s = item['study']; cal = s['sensor_calibration_sweeps']
    x, y = cal['ref_rosette'], cal['raw_rosette']
    clean = np.max(abs(y), axis=1) < 7
    for model in ('archived','pairwise','three_way','rational'):
        errors = []
        for j in np.flatnonzero(clean):
            train = clean & (np.arange(len(x)) != j)
            row = []
            for i in range(4):
                F = features(x, i, model)
                coef = np.linalg.lstsq(F[train], y[train,i], rcond=None)[0]
                row.append(F[j] @ coef - y[j,i])
            errors.append(row)
        e = np.asarray(errors)
        print(model, 'LOO worst channel RMSE', np.sqrt(np.mean(e**2,axis=0)).max(), 'max abs', abs(e).max())
    co = m._rosette_coefficients(x, y)
    sweep = cal['single_bank_sweeps']
    names = list(sweep)
    print('single-bank sweep keys:', names)
    bank, ref, raw = (np.asarray(sweep[k]) for k in names)
    t = np.zeros((len(bank),4)); t[np.arange(len(bank)),bank.astype(int)] = ref
    pred = np.array([m._features(q,i) @ co[i] for q,i in zip(t,bank.astype(int))]); e = abs(raw-pred)
    print('single-bank independent max error (excl. gross fault)', e[abs(raw)<7].max(), 'gross fault minimum error', e[abs(raw)>=7].min() if (abs(raw)>=7).any() else None)
    S, _ = m._forward(s); Z = s['Z_nuisance']; P = np.eye(len(S))-Z@np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
    def worst_slice(e):
        return np.sqrt(np.mean(e.reshape(len(s['mu_shifts']),-1,4)**2,axis=1)).max()
    for j, pulse in enumerate(item.get('campaign_diagnostics') or []):
        e = m._calibrate(dict(s,y_obs_raw=pulse['y_pulse_raw']))-S@pulse['w_pulse']
        print('pulse',j,'worst shift-bank RMSE unprofiled/profiled',worst_slice(e),worst_slice(P@e))
        assert worst_slice(e) < 2e-5 and worst_slice(P@e) < 2e-5
