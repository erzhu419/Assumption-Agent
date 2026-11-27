import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f:
    s = pickle.load(f)[0]['study']
c = s['sensor_calibration_sweeps']
x,y = c['ref_rosette'],c['raw_rosette']
z = np.column_stack([np.ones(len(x)),x])
# Hold out the largest amplitudes together, rather than randomly splitting them across folds.
folds = np.array_split(np.argsort(np.max(abs(x),axis=1)),5)
errors = {k: np.empty_like(y) for k in ['full affine','bank affine','saturating rational']}
for te in folds:
    tr = np.setdiff1d(np.arange(len(x)),te)
    full = np.linalg.lstsq(z[tr],y[tr],rcond=None)[0]
    errors['full affine'][te] = z[te]@full-y[te]
    for j in range(4):
        zz = z[:,[0,j+1]]
        d = np.linalg.lstsq(zz[tr],y[tr,j],rcond=None)[0]
        errors['bank affine'][te,j] = zz[te]@d-y[te,j]
        def pred(p,ix):
            a = p[-1]
            return (z[ix]@p[:-1])/(1+a*x[ix,j]**2)
        p = least_squares(lambda p: pred(p,tr)-y[tr,j],np.r_[full[:,j],0.],bounds=(np.r_[np.full(5,-np.inf),0],np.full(6,np.inf)),gtol=1e-12).x
        errors['saturating rational'][te,j] = pred(p,te)-y[te,j]
for k,e in errors.items():
    print(k,'pooled RMSE',np.sqrt(np.mean(e**2)),'worst amplitude-fold/bank RMSE',max(np.sqrt(np.mean(e[te]**2,axis=0)).max() for te in folds))
# The independently acquired single-bank sweeps can expose cross-bank bias.
full = np.linalg.lstsq(z,y,rcond=None)[0]
bk = c['single_bank_sweeps']; j = bk['bank_id']; r = bk['ref_signal']
e = full[0,j] + full[j+1,j]*r - bk['raw_readout']
print('Independent sweep RMSE by bank',[np.sqrt(np.mean(e[j==k]**2)) for k in range(4)])
# Compare nuisance profiling and omitting nuisance with identical shift-disjoint folds.
d,_ = m._design(s); yy = m._calibrate(s); zz = s['Z_nuisance']
shift = np.arange(len(yy))//len(s['sensor_ij']); banks = s['row_bank_id']; ridge=s['source_ridge']
for profile in (False,True):
    e = np.empty_like(yy)
    for h in np.unique(shift):
        tr = shift != h; te = ~tr
        Z = zz[tr]; P = np.linalg.solve(Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]),Z.T) if profile else np.zeros((Z.shape[1],len(Z)))
        a = d[tr] - Z@(P@d[tr]); b = yy[tr] - Z@(P@yy[tr])
        w,_ = m._box_qp(a.T@a+ridge*np.eye(a.shape[1]),-a.T@b,s['u_box'],s['total_budget'])
        e[te] = d[te]@w+zz[te]@(P@(yy[tr]-d[tr]@w))-yy[te]
    print('Profile nuisance',profile,'worst shift/bank RMSE',max(np.sqrt(np.mean(e[(shift==h)&(banks==k)]**2)) for h in np.unique(shift) for k in np.unique(banks)))
