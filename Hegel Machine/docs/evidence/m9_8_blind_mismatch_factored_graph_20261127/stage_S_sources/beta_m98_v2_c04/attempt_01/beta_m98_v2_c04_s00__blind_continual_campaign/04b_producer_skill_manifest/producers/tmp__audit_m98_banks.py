import pickle, runpy
import numpy as np
m = runpy.run_path('/holdout_task/batch_solver.py')
item = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s = item['study']; c = s['sensor_calibration_sweeps']; x,y = c['ref_rosette'],c['raw_rosette']
D = np.c_[np.ones(len(x)),x]; C = np.linalg.lstsq(D,y,rcond=None)[0]
print('affine intercept', C[0], 'mixing perturbation from diagonal', C[1:]-np.diag(np.diag(C[1:])))
for k,v in c['single_bank_sweeps'].items():
    a = np.asarray(v); print('single-bank',k,'shape',a.shape,'first rows',a[:2])
obs = m['_calibrate'](s).reshape(-1,4)
print('observations outside calibration reference range by channel', ((obs<x.min(0))|(obs>x.max(0))).sum(0), 'of',len(obs))
S,_ = m['_forward'](s); Z=s['Z_nuisance']; target=obs.ravel(); n=len(s['sensor_ij']); B=s['total_budget']; cap=s['u_box']; ridge=s['source_ridge']
banks=np.asarray(s['row_bank_id']).ravel(); groups=np.tile(np.arange(n)//4,len(s['mu_shifts'])); errs={}
for profile in (False,True):
    e=np.empty_like(target)
    for group in np.unique(groups):
        test=groups==group; A=S[~test]; b=target[~test]; z=Z[~test]
        if profile:
            p=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T); X=A-z@(p@A); q=b-z@(p@b)
        else:
            X,q=A,b
        w,_=m['_box_budget'](X.T@X+ridge*np.eye(X.shape[1]),-X.T@q,cap,B)
        pred=S[test]@w
        if profile: pred += Z[test]@(p@(b-A@w))
        e[test]=pred-target[test]
    errs[profile]=e
    print('profile',profile,'pooled RMSE',np.sqrt(np.mean(e**2)),'bank RMSE',[(int(k),float(np.sqrt(np.mean(e[banks==k]**2)))) for k in np.unique(banks)])
