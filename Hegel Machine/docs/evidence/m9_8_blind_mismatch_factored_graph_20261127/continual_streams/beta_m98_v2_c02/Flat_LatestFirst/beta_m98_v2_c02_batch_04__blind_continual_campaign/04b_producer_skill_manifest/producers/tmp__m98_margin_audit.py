import importlib.util
import pickle
from itertools import combinations_with_replacement
import numpy as np
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f:
    s=pickle.load(f)[0]['study']
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']
# The recorded gross corruptions are separated by more than seven raw units.
# Score on real retained measurements, but leave the corrupt measurements in training.
clean=np.max(abs(y),axis=1)<9
ids=np.flatnonzero(clean)
axis=np.count_nonzero(x,axis=1)==1
mixed=ids[~axis[ids]]
outer=[('single-axis',ids[axis[ids]])]+[(f'mixed amplitude {i}',v) for i,v in enumerate(np.array_split(mixed[np.argsort(np.linalg.norm(x[mixed],axis=1))],4))]
terms=[()] + [t for d in (1,2,3) for t in combinations_with_replacement(range(4),d)]
full=np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
for label,test in outer:
    train=np.ones(len(x),dtype=bool); train[test]=False
    # Omit all duplicate single-bank sweeps in this audit to prevent holdout leakage.
    sub=dict(c,ref_rosette=x[train],raw_rosette=y[train],single_bank_sweeps={k:v[:0] for k,v in c['single_bank_sweeps'].items()})
    coef=m._fit_rosette(sub)
    compact=m._rosette_raw(x[test],coef)
    ref=train & clean
    cubic=full[test]@np.linalg.lstsq(full[ref],y[ref],rcond=None)[0]
    # Rational challengers get the benefit of oracle screening of training corruption.
    q=np.stack([x,x-np.roll(x,-1,axis=1),x+np.roll(x,-1,axis=1)])
    rat=[]
    for qq in q:
        for a in (0.001,0.01,0.1,1.):
            preds=[]
            for j in range(4):
                B=m._rosette_basis(x,j)/(1+a*qq[:,j,None]**2)
                preds.append(B[test]@np.linalg.lstsq(B[ref],y[ref,j],rcond=None)[0])
            rat.append(np.column_stack(preds))
    worst=lambda pred: np.max(np.sqrt(np.mean((pred-y[test])**2,axis=0)))
    print(label,'compact',worst(compact),'full cubic',worst(cubic),'best rational (optimistic)',min(map(worst,rat)))
    assert worst(compact)<5e-5
S,_=m._forward(s); y=m._calibrate(s); Z=s['Z_nuisance']; k=Z.shape[1]; ns=len(s['sensor_ij'])
for j,mu in enumerate(s['mu_shifts']):
    test=np.arange(len(y))//ns==j
    a,b,z=S[~test],y[~test],Z[~test]
    p=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(k),z.T)
    A=a-z@(p@a); B=b-z@(p@b)
    ridge=s['source_ridge']*np.eye(a.shape[1]); cap,total=s['u_box'],s['total_budget']
    w,_=m._box_budget_qp(A.T@A+ridge,A.T@B,cap,total)
    w0,_=m._box_budget_qp(a.T@a+ridge,a.T@b,cap,total)
    pred=S[test]@w+Z[test]@(p@(b-a@w))
    rms=lambda a: np.sqrt(np.mean((a-y[test])**2))
    print('held shift',mu,'nuisance profiled RMSE',rms(pred),'nuisance omitted RMSE',rms(S[test]@w0))
    assert rms(pred)<0.04
