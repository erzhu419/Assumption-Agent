import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; Y=tr['traj']; np.set_printoptions(precision=5,suppress=True,linewidth=120)
print('times telemetry',t,s['telemetry_vec']); print('svd',np.linalg.svd(Y,compute_uv=False))
u=s['panel_a']['u'];v=s['panel_a']['v']; H=np.c_[np.ones(len(u)),u]; k=np.ones(len(u),bool)
for i in range(5):
 A=np.linalg.lstsq(H[k],v[k],rcond=None)[0]; r=np.linalg.norm(v-H@A,axis=1); k=r<max(4*np.median(r),.002)
print('A',A,'residuals',r)
for m in [1,2,3,4]:
 def calc(z,ret=False):
  q=-np.expm1(-t[:,None]*np.exp(z)); a=np.linalg.lstsq(q,Y.T,rcond=None)[0].T; return (a,q) if ret else (a@q.T-Y).ravel()
 opt=least_squares(calc,np.log(np.geomspace(.25,3,m))); print('rates',np.exp(opt.x),'rmse',np.mean(opt.fun**2)**.5); a,q=calc(opt.x,True)
 if m==2: np.savez('/tmp/amp.npz',a=a,q=q,rates=np.exp(opt.x)); print('amplitude ranges',a.min(0),a.max(0),'corr with y',np.corrcoef(np.c_[tr['y'],a].T))
u=s['panel_c']['u']; U=u.T; d={'one':np.ones(len(u))}
for i,a in enumerate(U):
 for k in range(1,5): d[f'u{i}^{k}']=a**k
for k in [2,3,4]:
 for ids in combinations(range(4),k):d['prod'+str(ids)]=U[list(ids)].prod(0)
for i,j in combinations(range(4),2):
 a,b=U[[i,j]]
 for sign in (-1,1):
  for f in (.5,1,1.5,2,3,4):
   w=f*(a+sign*b)
   for name,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:d[f'{name} {f} ({i},{sign*j})']=fun(w)
X=np.column_stack(list(d.values())); names=np.array(list(d));
for k in ['v1','v2']:
 for y in s['panel_c'][k].T:
  corr=abs(np.corrcoef(X[:,1:].T,y)[-1,:-1]); ix=np.argsort(-corr)[:5]; print(k,list(zip(names[ix+1],np.round(corr[ix],3))))
