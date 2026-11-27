import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('candidate', root + 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]
s = item['study']
ref = s['sensor_calibration_sweeps']['ref_rosette']
raw = s['sensor_calibration_sweeps']['raw_rosette']

def features(x, i, extended):
    base = m._rosette_terms(x, i)
    if not extended:
        return base
    a = x[:, i]
    b = x[:, (i+1)%4]
    c = x[:, (i-1)%4]
    d = x[:, (i+2)%4]
    return np.column_stack((base, d, a*b*c, (a-b)/(1+(a-b)**2), (a+b)/(1+(a+b)**2)))

# Hold out contiguous amplitude bands, including the strongest sweep band.
folds = np.array_split(np.argsort(np.max(np.abs(ref), axis=1)), 5)
for label, ext in [('current', False), ('extended', True)]:
    worst = 0.
    for j, test in enumerate(folds):
        train = np.concatenate([f for i,f in enumerate(folds) if i != j])
        x, y = ref[train], raw[train]
        coeff = m._fit_rosette(x, y)
        fitted = np.column_stack([features(x,i,False)@coeff[:,i] for i in range(4)])
        rowerr = np.linalg.norm(y-fitted,axis=1)
        keep = rowerr <= max(10*np.median(rowerr),1e-5)
        if ext:
            coeff = np.column_stack([np.linalg.lstsq(features(x[keep],i,True),y[keep,i],rcond=None)[0] for i in range(4)])
        pred = np.column_stack([features(ref[test],i,ext)@coeff[:,i] for i in range(4)])
        residual = raw[test]-pred
        # Report gross discrepancies separately instead of letting them mask the ordinary sweeps.
        gross = np.max(np.abs(residual),axis=1)>0.05
        rmse = np.sqrt(np.mean(residual[~gross]**2,axis=0))
        worst = max(worst,rmse.max())
        print(label,'band',j,'ordinary channel RMSE',rmse,'gross rows',gross.sum())
    print(label,'worst ordinary band/channel RMSE',worst)

coef=m._fit_rosette(ref,raw)
for label, trace in [('observation',s['y_obs_raw'])]+[(f'pulse{i}',p['y_pulse_raw']) for i,p in enumerate(item['campaign_diagnostics'] or [])]:
    x=m._calibrate(dict(s,y_obs_raw=trace)).reshape(-1,4)
    f=np.column_stack([features(x,i,False)@coef[:,i] for i in range(4)])
    err=np.max(np.abs(f-trace.reshape(-1,4)))
    print(label,'maximum inverse round-trip error',err,'maximum calibrated amplitude',np.abs(x).max())
    assert err < 1e-8
print('study fields',sorted(s.keys()))
