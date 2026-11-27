import pickle, numpy as np, importlib.util
from itertools import product

def archived(batch):
    path = f'/tmp/hegel_continual_archive/beta_m99_c02_batch_{batch:02d}_solver.py'
    spec = importlib.util.spec_from_file_location(f'archive{batch}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
c = s['sensor_calibration_sweeps']
x, y = c['ref_rosette'], c['raw_rosette']
a, p = archived(4), archived(2)

def evaluate(rows, fit):
    return np.stack([a._sensor(row, fit)[0] for row in rows])

f = a._fit_calibration(x, y)
print('robust cubic coefficients', f.round(5))
err = np.empty_like(y)
for fold in range(5):
    test = np.arange(len(x)) % 5 == fold
    err[test] = evaluate(x[test], a._fit_calibration(x[~test], y[~test])) - y[test]
print('OOF sweep abs residual q50,q80,q90', np.quantile(abs(err), [.5,.8,.9], axis=0).round(6))
q = c['single_bank_sweeps']; bank = q['bank_id']; ref = q['ref_signal']; raw = q['raw_readout']
rows = np.zeros((len(bank), 4)); rows[np.arange(len(bank)), bank] = ref
for j in range(4):
    take=bank==j
    print('single bank', j, np.c_[ref[take], raw[take], (evaluate(rows, f)[np.arange(len(bank)),bank]-raw)[take]].round(5).tolist())
print('bank pattern', s['row_bank_id'][:16]); print('obs raw ranges',s['y_obs_raw'].reshape(-1,4).min(0),s['y_obs_raw'].reshape(-1,4).max(0))
for modes in product(('open','reflect','period'), repeat=2):
    axes=[]
    for axis,n,mode in zip('xy',(s['nx'],s['ny']),modes):
        args=(s['kappa_'+axis],s['vel_'+axis],s['L'+axis]/(n-1))
        axes.append(p._axis(*args) if mode=='period' else a._axis(*args,s['robin_'+axis] if mode=='open' else 0.))
    S,_=a._responses(s,axes)
    diffs=np.array([evaluate((S@pulse['w_pulse']).reshape(-1,4),f)-pulse['y_pulse_raw'].reshape(-1,4) for pulse in item['campaign_diagnostics']])
    print(modes,'pulse RMSE', np.sqrt(np.mean(diffs**2,axis=(1,2))).round(7),'MAE',np.mean(abs(diffs),axis=(1,2)).round(7),'bank RMSE',np.sqrt(np.mean(diffs**2,axis=(0,1))).round(7))
