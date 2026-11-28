import importlib.util
import json
import pickle
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
a = json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
print('diagnostics:', type(item.get('campaign_diagnostics')).__name__, 'apparatus metadata:', [{k: v for k,v in d.items() if 'trace' not in k and len(str(v)) < 400} for d in a])
c = s['sensor_calibration_sweeps']; x, y = c['ref_rosette'], c['raw_rosette']; single = c['single_bank_sweeps']
print('single-bank sweep schema:', type(single).__name__, [(str(k), np.shape(v) if not isinstance(v,dict) else list(v)) for k,v in single.items()] if isinstance(single,dict) else str(single)[:300])
# Fit both candidate calibration grammars inside each held-out sweep split.
P = np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
for label, D in [('affine', np.c_[np.ones(len(x)),x]), ('pairwise quadratic', np.c_[np.ones(len(x)),x,P]), ('channel saturation', np.c_[np.ones(len(x)),x,x/(1+x*x)])]:
    E=[]
    for i in range(len(x)):
        f=np.arange(len(x))!=i
        E.append(y[i]-D[i]@np.linalg.lstsq(D[f],y[f],rcond=None)[0])
    print('sweep',label,'worst channel LOO RMSE',np.sqrt(np.mean(np.square(E),axis=0)).max())

base, _ = m._transport(s)
y = m._calibrate(s); Z = s['Z_nuisance']; cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']; nr=s['nuisance_ridge']
N = len(s['sensor_ij']); bank=np.asarray(s['row_bank_id']); groups=np.arange(len(y))%N//4

def predict(S, train, nuisance=True):
    A=S[train]; b=y[train]; q=Z[train]
    if nuisance:
        T=np.linalg.solve(q.T@q+nr*np.eye(q.shape[1]), q.T)
        A=A-q@(T@A); b=b-q@(T@b)
    w,_=m._box_simplex(A.T@A+ridge*np.eye(A.shape[1]),-A.T@b,cap,budget)
    gamma=T@(y[train]-S[train]@w) if nuisance else np.zeros(q.shape[1])
    return S@w+Z@gamma

def forward(p):
    return m._transport(dict(s, robin_x=s['robin_x']*p[0], robin_y=s['robin_y']*p[1]))[0]

def rms(e): return np.sqrt(np.mean(e**2))

def validate(kind):
    E=np.zeros(len(y)); vals=[]
    for g in np.unique(groups):
        train=groups!=g
        S=base
        if kind=='fit wall + nuisance':
            p=least_squares(lambda p:(predict(forward(p), train)-y)[train], np.ones(2), bounds=(.25,4), max_nfev=35).x
            vals.append(np.round(p,3).tolist()); S=forward(p)
        E[~train]=(predict(S,train,kind!='fixed wall, no nuisance')-y)[~train]
    print(kind,'pooled RMSE',rms(E),'worst spatial rosette RMSE',max(rms(E[groups==g]) for g in np.unique(groups)),'bank RMSE',[(int(k),round(rms(E[bank==k]),6)) for k in np.unique(bank)],'fold walls',vals)

for kind in ('fixed wall, no nuisance','fixed wall + nuisance','fit wall + nuisance'):
    validate(kind)
