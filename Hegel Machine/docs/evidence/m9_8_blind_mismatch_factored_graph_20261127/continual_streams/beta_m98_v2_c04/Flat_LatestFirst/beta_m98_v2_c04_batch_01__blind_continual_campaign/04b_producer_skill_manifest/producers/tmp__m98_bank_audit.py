import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; S,_=m._forward(s); y=m._calibrate(s); Z=s['Z_nuisance']; banks=np.asarray(s['row_bank_id']).ravel(); k=len(s['mu_shifts']); n=len(y)//k
print('bank layout',banks.shape,'rows per shift',n,'bank counts',np.unique(banks,return_counts=True)); print('single-bank sweep type:',type(s['sensor_calibration_sweeps']['single_bank_sweeps']).__name__)
if banks.size==n: banks=np.tile(banks,k)
assert banks.size==len(y)
from scipy.linalg import solve
A=Z.T@Z+s['nuisance_ridge']*np.eye(Z.shape[1]); P=solve(A,Z.T,assume_a='pos'); X=S-Z@(P@S); r=y-Z@(P@y); cap=s['u_box']; budget=s['total_budget']; ridge=s['source_ridge']; I=np.eye(S.shape[1]); full=m.solve_batch(s)['weights']
print('full projected residual RMS:',np.sqrt(np.mean((X@full-r)**2)))
pooled=[]; sliced=[]
for b in np.unique(banks):
 test=banks==b; train=~test; T=Z[train]; Xt=S[train]; yt=y[train]; prof=solve(T.T@T+s['nuisance_ridge']*np.eye(T.shape[1]),T.T,assume_a='pos'); xp=Xt-T@(prof@Xt); yp=yt-T@(prof@yt)
 w,_=m._box_budget(xp.T@xp+ridge*I,-xp.T@yp,cap,budget); w0,_=m._box_budget(Xt.T@Xt+ridge*I,-Xt.T@yt,cap,budget)
 err=y[test]-S[test]@w-Z[test]@(prof@(yt-Xt@w)); err0=y[test]-S[test]@w0
 # Compare centered errors as well as RMSE so a bank offset cannot mask shape error.
 rm=lambda x:float(np.sqrt(np.mean(x*x)))
 print('bank',b,'holdout RMSE profiled/unprofiled',rm(err),rm(err0),'centered',rm(err-err.mean()),rm(err0-err0.mean()),'weight change',np.max(abs(w-full)))
