import pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y,T=(d[k] for k in ('x','y','traj')); rng=np.random.default_rng(53)
folds=[np.arange(72),*[rng.permutation(72) for _ in range(4)],np.argsort(np.max(abs(x-np.median(x,axis=0)),axis=1)),np.argsort(abs(y-np.median(y)))];
bases=[[.573,2.265,-.092,.0086,9.92,.004,13.27],[11.86,1.101,-.02345,.001585,8.83,.000741,12.64,-.000136,20.25],[13.86,1.09,-.02049,.001375,8.864,.000629,12.68,-.000122,20.32,.000055,28.04],[7.59,2.355,-.09492,.00312,9.631,.000627,13.545,-.00005,21.5,.00966,5.90]]
for pair in [('core',.7,bases[:1]),('core-low',.25,bases[:1]),('three',.5,bases[1:2]),('four-high',.5,bases[2:3]),('four-low',.5,bases[3:])]:
 name,weight,starts=pair; out=[]
 for idx in folds:
  i,j=idx[:56],idx[56:]; sub=dict(s,train={**d,**{k:d[k][i] for k in ('x','y','traj')}}); f,p,cy=_fit(sub,weight,starts); yp,tp=f(x[j]); out.append([np.mean(abs(yp-y[j])),np.mean(abs(tp-T[j])),np.max(np.mean(abs(tp-T[j]),axis=0))]);
  if len(out)==1:print(name,'params',p,'link',cy,flush=True)
 print(name,np.round(out,6),flush=True)
t=s['eval_times']; b=np.c_[np.log1p(1.41246*t),-np.expm1(-.69078*t)]; r=T-np.linalg.lstsq(b,T.T,rcond=None)[0].T@b.T; print('trajectory orthogonal residual',np.std(r),np.max(abs(r)),np.linalg.svd(r,compute_uv=False))
