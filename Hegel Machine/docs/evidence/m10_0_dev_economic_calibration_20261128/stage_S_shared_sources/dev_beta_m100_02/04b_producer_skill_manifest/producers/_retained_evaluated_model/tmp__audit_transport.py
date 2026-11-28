import importlib.util, json, pathlib, pickle
import numpy as np
p = pathlib.Path('/holdout_task')
spec = importlib.util.spec_from_file_location('batch_solver', p/'batch_solver.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = pickle.loads((p/'public_batch.pkl').read_bytes())[0]['study']
c = s['sensor_calibration_sweeps']; X = c['ref_rosette']; Y = c['raw_rosette']
def feat(x, kind):
    f = [np.ones(len(x)), *x.T]
    if kind == 'interaction':
        f += [x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]
        f += [x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i,4) for k in range(j,4)]
    elif kind != 'affine':
        a = float(kind)
        f = [np.ones(len(x)), *(x/(1+a*x*x)).T, *((x[:,i]-x[:,j])/(1+a*(x[:,i]+x[:,j])**2) for i in range(4) for j in range(i+1,4))]
    return np.asarray(f).T
rng = np.random.default_rng(39)
# Axis-only sweeps and simultaneous sweeps are represented in each fold.
sets = [np.flatnonzero(np.count_nonzero(X,axis=1)<=1),np.flatnonzero(np.count_nonzero(X,axis=1)>1)]
folds = [np.concatenate(t) for t in zip(*(np.array_split(rng.permutation(a),5) for a in sets))]
def err(kind,train,test):
    A = feat(X[train],kind)
    return Y[test]-feat(X[test],kind)@np.linalg.lstsq(A,Y[train],rcond=None)[0]
allrows = np.arange(len(X))
scores = {k: [] for k in ('affine','interaction','rational nested')}
for i,test in enumerate(folds):
    train = np.setdiff1d(allrows,test)
    pool = ['0.000001','0.00001','0.0001','0.001','0.01','0.1']
    best = min(pool,key=lambda k: sum(np.sum(err(k,np.setdiff1d(train,v),v)**2) for j,v in enumerate(folds) if j!=i))
    for label,k in [('affine','affine'),('interaction','interaction'),('rational nested',best)]:
        r=err(k,train,test); scores[label].append(float(np.max(np.sqrt(np.mean(r*r,axis=0)))))
print('worst calibration fold/channel RMSE', {k:max(v) for k,v in scores.items()})
b = c['single_bank_sweeps']; print('bank sweep RMS from identity', {int(k):float(np.sqrt(np.mean((b['ref_signal'][b['bank_id']==k]-b['raw_readout'][b['bank_id']==k])**2))) for k in np.unique(b['bank_id'])})
S,_=m._responses(s,m._operator(s)); y=m._calibrate(s); Z=s['Z_nuisance']; ids=s['row_bank_id']; n=len(s['sensor_ij']); rows=np.arange(len(y)); ridge=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']
# Hold out whole rosettes at each position across shifts to avoid channel leakage.
for profile in (False,True):
    residual=np.empty(len(y))
    for j in range(0,n,4):
        test=(rows%n>=j)&(rows%n<j+4); train=~test
        A=S[train]; q=y[train]; z=Z[train]
        if profile:
            P=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T)
            T=A-z@(P@A); r=q-z@(P@q)
        else:
            T=A; r=q
        w,_=m._box_simplex(T.T@T+ridge*np.eye(T.shape[1]),-T.T@r,cap,budget)
        residual[test]=S[test]@w-y[test]
        if profile:
            residual[test]+=Z[test]@(P@(q-A@w))
    print('nuisance profiled',profile,'OOF pooled RMSE',np.sqrt(np.mean(residual**2)),'bank RMSE',{int(k):float(np.sqrt(np.mean(residual[ids==k]**2))) for k in np.unique(ids)})
log=pathlib.Path('/tmp/hegel_continual_archive/archive_apparatus_log.json'); print('apparatus log:',log.read_text() if log.exists() else 'absent')
