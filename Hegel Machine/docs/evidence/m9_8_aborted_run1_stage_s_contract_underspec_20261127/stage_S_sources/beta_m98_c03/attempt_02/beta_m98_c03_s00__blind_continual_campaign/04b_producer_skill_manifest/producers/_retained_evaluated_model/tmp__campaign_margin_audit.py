import importlib.util
import itertools
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
y = raw - x
single = np.count_nonzero(x, axis=1) == 1
fold = np.arange(len(x)) % 4
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
poly = m._features(x, [()] + [t for k in (1,2,3) for t in itertools.combinations_with_replacement(range(4),k)])
def design(a, mode):
    q = x if mode == 'channel' else np.column_stack([(x[:,i] + sign*x[:,j])/np.sqrt(2) for i,j in pairs for sign in (-1,1)])
    return np.column_stack((np.ones(len(x)), x, q/(1+a*q*q)-q))
def fit_predict(P, train, test):
    return P[test] @ np.linalg.lstsq(P[train], y[train], rcond=None)[0]
options = [('identity', None), ('affine', poly[:,:5]), ('quadratic', poly[:,:15]), ('cubic', poly)]
options += [(f'{mode}@{a}', design(a,mode)) for mode in ('channel','pair') for a in (.1,.5,2.)]
base = y.copy(); nested = np.empty_like(y); selections = []
for k in range(4):
    train, test = fold != k, fold == k
    scores = []
    for name,P in options:
        errors = []
        for j in range(4):
            if j == k: continue
            val = fold == j
            errors.append(y[val] if P is None else y[val] - fit_predict(P, train & ~val, val))
        scores.append(np.mean(np.concatenate(errors)**2))
    name,P = options[np.argmin(scores)]
    selections.append(name)
    nested[test] = y[test] if P is None else y[test]-fit_predict(P,train,test)
for name,e in [('identity',base),('nested candidate library',nested)]:
    rms = np.sqrt(np.mean(e**2))
    worst = max(np.sqrt(np.mean(e[mask]**2,axis=0)).max() for mask in (single,~single))
    print(name, 'pooled RMSE',rms,'worst bank-by-sweep RMSE',worst)
print('outer-fold choices:',selections)
bank = c['single_bank_sweeps']
print('dedicated single-bank identity RMSE:',[float(np.sqrt(np.mean((bank['raw_readout'][bank['bank_id']==k]-bank['ref_signal'][bank['bank_id']==k])**2))) for k in range(4)])
cal=m._calibration(c)
p=s['reference_tracer_pulses']; ns=len(s['sensor_ij']); ng=len(s['mu_shifts'])
for label in ('nominal','held-out tracer'):
    metrics=[]
    for k,pulse in enumerate(p):
        t=dict(s,reference_tracer_pulses=None if label=='nominal' else [q for j,q in enumerate(p) if j!=k]); S,_=m._transport(t,cal)
        err=(S@pulse['w_pulse']-cal(pulse['y_pulse_raw'])).reshape(ng,ns)
        metrics.extend(np.sqrt(np.mean(err**2,axis=1)))
    print(label,'worst held-pulse shift RMSE',max(metrics))
