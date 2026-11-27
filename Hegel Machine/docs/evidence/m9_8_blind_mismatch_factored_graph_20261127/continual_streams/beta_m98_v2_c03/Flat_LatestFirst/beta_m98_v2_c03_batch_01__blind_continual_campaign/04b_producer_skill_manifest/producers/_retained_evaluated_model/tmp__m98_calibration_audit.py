import importlib.util
import json
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
items = pickle.load(open(root+'public_batch.pkl', 'rb'))
for rec in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
    print('apparatus entry:', {k:v for k,v in rec.items() if k != 'facility_telemetry_trace'})
for item in items:
    s = item['study']
    sweeps = s['sensor_calibration_sweeps']
    x, y = sweeps['ref_rosette'], sweeps['raw_rosette']
    pulses = item.get('campaign_diagnostics') or []
    S, _ = m._forward(s)
    xp = np.concatenate([(S @ p['w_pulse']).reshape(-1,4) for p in pulses])
    yp = np.concatenate([p['y_pulse_raw'].reshape(-1,4) for p in pulses])
    def basis(t, model):
        a = np.c_[np.ones(len(t)), t]
        q = np.column_stack([t[:,i]*t[:,j] for i in range(4) for j in range(i,4)])
        pairs = np.column_stack([t[:,i]+sign*t[:,j] for i in range(4) for j in range(i+1,4) for sign in (-1,1)])
        if model == 'affine': return a
        if model == 'quadratic': return np.c_[a, q]
        u = np.c_[t,pairs]
        return np.c_[a, u**3/(1+u**2)]
    models = ('affine','quadratic','saturating')
    ids = np.random.default_rng(81).permutation(len(x))
    fold = np.empty(len(x),int)
    fold[ids] = np.arange(len(x))%5
    def evaluate(train, test, model):
        return basis(x[test],model) @ np.linalg.lstsq(basis(x[train],model), y[train],rcond=None)[0] - y[test]
    nested = np.empty_like(y)
    selected = []
    for f in range(5):
        train, test = fold!=f, fold==f
        scores = [np.mean(np.concatenate([evaluate(train & (fold!=g),fold==g,model) for g in range(5) if g!=f])**2) for model in models]
        selected.append(models[np.argmin(scores)])
        nested[test] = evaluate(train,test,selected[-1])
    print('nested selections', selected, 'worst channel RMSE', np.sqrt(np.mean(nested**2,axis=0)).max())
    for model in models:
        e = np.concatenate([evaluate(fold!=f,fold==f,model) for f in range(5)])
        q = np.linalg.lstsq(basis(x,model),y,rcond=None)[0]
        ep = (basis(xp,model) @ q-yp).reshape(-1,len(s['mu_shifts']),len(s['sensor_ij'])//4,4)
        print(model, 'OOF sweep channel RMSE', np.sqrt(np.mean(e**2,axis=0)), 'worst pulse/shift/channel raw RMSE', np.sqrt(np.mean(ep**2,axis=2)).max())
