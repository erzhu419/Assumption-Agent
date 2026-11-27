import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
st = item['study']
sw = st['sensor_calibration_sweeps']
x, y = sw['ref_rosette'], sw['raw_rosette']
scale = np.std(x, axis=0)

def features(x, model, scale):
    cols = [np.ones(len(x)), *x.T]
    if model == 'quadratic':
        cols += [x[:, i]*x[:, j] for i in range(4) for j in range(i, 4)]
    if model == 'rational':
        cols += [x[:, i]/(1+(x[:, i]/scale[i])**2) for i in range(4)]
        cols += [(x[:, i]-x[:, j])/(1+((x[:, i]-x[:, j])/np.hypot(scale[i],scale[j]))**2) for i in range(4) for j in range(i+1,4)]
    return np.column_stack(cols)

folds = np.array_split(np.random.default_rng(98).permutation(len(x)), 5)
for name in ('affine', 'quadratic', 'rational'):
    out = np.empty_like(y)
    worst = 0
    for test in folds:
        train = np.setdiff1d(np.arange(len(x)),test)
        scale = np.std(x[train],axis=0)
        coef = np.linalg.lstsq(features(x[train],name,scale),y[train],rcond=None)[0]
        out[test] = features(x[test],name,scale)@coef-y[test]
        worst = max(worst,np.max(np.sqrt(np.mean(out[test]**2,axis=0))))
    print(name,'OOF RMS by channel',np.sqrt(np.mean(out**2,axis=0)),'worst fold/channel RMS',worst)
S, _ = m._forward(st)
Z = st['Z_nuisance']
B = np.asarray(st['row_bank_id'])
for i,p in enumerate(item['campaign_diagnostics']):
    r = m._calibrate(dict(st,y_obs_raw=p['y_pulse_raw']))-S@p['w_pulse']
    q = r-Z@np.linalg.solve(Z.T@Z+st['nuisance_ridge']*np.eye(Z.shape[1]),Z.T@r)
    print('pulse',i,'raw/profiled RMS',np.sqrt(np.mean(r*r)),np.sqrt(np.mean(q*q)))
    print('pulse',i,'shift RMS',np.sqrt(np.mean(r.reshape(len(st['mu_shifts']),-1)**2,axis=1)))
    print('pulse',i,'banks: count, signed mean, RMS',[(int(b),int((B==b).sum()),float(r[B==b].mean()),float(np.sqrt(np.mean(r[B==b]**2)))) for b in np.unique(B)])
print('single-bank sweep shapes', {k: np.shape(v) for k,v in sw['single_bank_sweeps'].items()})
