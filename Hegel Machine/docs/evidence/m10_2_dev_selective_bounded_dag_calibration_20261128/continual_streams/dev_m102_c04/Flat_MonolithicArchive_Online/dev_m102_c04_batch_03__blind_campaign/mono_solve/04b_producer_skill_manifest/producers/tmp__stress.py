import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import _calibration,_panel_operators,_affine as aff,_ls as ls,solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; q=_calibration(s['panel_a'])(x); f,g=_panel_operators(s['panel_c']); p,r=f(q),g(q); A=aff(np.c_[p,r,np.sum(p*r,axis=1)]); O=np.column_stack([p[:,i]*r[:,j] for i in range(4) for j in range(4) if i!=j]); pairs=[(i,j) for i in range(4) for j in range(4) if i!=j]
rng=np.random.default_rng(42); order=np.argsort(np.linalg.norm(q,axis=1)); folds=[np.arange(52,72),order[-20:],order[:20:]]+[a for a in np.array_split(rng.permutation(72),4)]
allerr=[]
for J in folds:
 I=np.setdiff1d(np.arange(len(x)),J); Ai=A[I]; zz=O[I]-Ai@ls(Ai,O[I]); ty=T[I]-Ai@ls(Ai,T[I]); idx=np.argmax(np.sum((zz.T@ty)**2,axis=1)/np.sum(zz**2,axis=0)); B=np.c_[A,O[:,idx]]; Bi=B[I]; weights=ls(Bi,T[I]); fitted=Bi@weights
 pred=solve_batch(dict(s,train={k:v[I] if k in ('x','y','traj') else v for k,v in s['train'].items()},test={'x':x[J]})); maes=[np.mean(abs(pred['y_hat']-y[J])),np.mean(abs(pred['state_pred']-T[J])),np.mean(abs(A[J]@ls(Ai,y[I])-y[J])),np.mean(abs(B[J]@weights-T[J]))]
 for kind in ('rates','rates_const'):
  def basis(k):
   E=-np.expm1(-t[:,None]*k)
   return E if kind=='rates' else np.c_[E,np.ones(len(t))]
  def err(k):
   E=basis(k);return (E@ls(E,fitted.T)-fitted.T).ravel()
  rate=least_squares(err,[1.8,.42],bounds=([.8,.03],[8,.79])).x; E=basis(rate); pred=B[J]@ls(Bi,T[I]@np.linalg.pinv(E.T))@E.T
  maes.append(np.mean(abs(pred-T[J]))); print(kind,rate,end='; ')
 allerr.append(maes);print('holdout',len(J),'interaction',pairs[idx],'MAEs',np.round(maes,6))
print('worsts archive-y,archive-state,compact-y,unconstrained-state,projected-state,projected+constant-state',np.max(allerr,0))
