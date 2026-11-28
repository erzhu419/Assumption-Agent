import sys,pickle,itertools,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=m._calibrate(s); Z=s['Z_nuisance']; n=len(s['sensor_ij']); cap=s['u_box']; budget=s['total_budget'];
print('calibrated range',y.min(),y.max(),'shape',s['source_modes'].shape,'grid',s['nx'],s['ny'])
c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; r=c['raw_rosette']; good=np.max(abs(r),axis=1)<8; B=c['single_bank_sweeps']; ib=B['bank_id']; bb=np.eye(4)[ib]*B['ref_signal'][:,None]; rb=B['raw_readout'];
for family in ['mixed','axis','bank']:
 errors=[]
 for k in range(4):
  gb=np.flatnonzero((ib==k)&(abs(rb)<8)); axis=(np.count_nonzero(x,axis=1)==1); ids=np.flatnonzero(good & (axis if family=='axis' else ~axis)); X=m._sensor_features(x,k); Xb=m._sensor_features(bb,k)
  for i in gb if family=='bank' else ids:
   train=np.flatnonzero(good); b=gb
   if family=='bank': b=b[b!=i]
   else: train=train[train!=i]
   fit=np.linalg.lstsq(np.r_[X[train],Xb[b]],np.r_[r[train,k],rb[b]],rcond=None)[0]; errors.append((Xb[i]@fit-rb[i]) if family=='bank' else (X[i]@fit-r[i,k]))
 print('calibration LOO',family,'rms',np.sqrt(np.mean(np.square(errors))),'max',max(abs(np.array(errors))))
def fit(S,sub):
 A=S[sub]; z=Z[sub]; obs=y[sub]; D=np.c_[A,z]; Q=D.T@D; f=-D.T@obs; p=A.shape[1]; H=Q[:p,:p]; J=Q[p:,p:]; C=Q[:p,p:]; M=np.linalg.solve(J,C.T); b=np.linalg.solve(J,f[p:]); w,nu=m._box_simplex(H-C@M+1e-10*np.eye(p),f[:p]-C@b,cap,budget); g=-b-M@w; return w,g
ij=s['sensor_ij']; boundary=(ij[:,0]==0)|(ij[:,0]==s['nx']-1)|(ij[:,1]==0)|(ij[:,1]==s['ny']-1); allrows=np.arange(len(y));
folds=[allrows[allrows//n==k] for k in range(len(s['mu_shifts']))]+[np.flatnonzero(np.tile(boundary,len(s['mu_shifts'])))]+[np.flatnonzero((allrows%n)==k) for k in range(n)]
def res(walls):
 S,_=m._forward(s,walls); w,g=fit(S,allrows); return S@w+Z@g-y
scores=[]
for walls in itertools.product([0,1],repeat=4):
 S,_=m._forward(s,walls); w,g=fit(S,allrows); e=[]
 for test in folds:
  tr=np.setdiff1d(allrows,test); a,b=fit(S,tr); e.append(np.sqrt(np.mean((S[test]@a+Z[test]@b-y[test])**2)))
 scores.append((np.sqrt(np.mean((S@w+Z@g-y)**2)),walls,*e[:4],max(e[4:])))
print('columns: fit-rms walls held-shift x3 walls-held worst-sensor');
for v in sorted(scores): print(v[1],np.round([v[0],*v[2:]],6))
f=least_squares(res,np.full(4,.5),bounds=(-.2,3)); print('continuous relative Robin factors',f.x,'rms',np.sqrt(np.mean(f.fun**2)))
