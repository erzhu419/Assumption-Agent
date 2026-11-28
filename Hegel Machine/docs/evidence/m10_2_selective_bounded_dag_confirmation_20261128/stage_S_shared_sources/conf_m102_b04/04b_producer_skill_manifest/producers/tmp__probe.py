import sys,pickle,numpy as np
from math import factorial
from sklearn.preprocessing import PolynomialFeatures
sys.path.insert(0,'/holdout_task'); from batch_solver import solve_batch
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; m=np.c_[y,z]; np.set_printoptions(precision=4,suppress=True)
B=np.linalg.lstsq(np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'],rcond=None)[0]; xb=np.c_[np.ones(72),x]@B
for name,xx in [('raw',x),('B',xb)]:
 p=PolynomialFeatures(3); X=p.fit_transform(xx); W=np.linalg.lstsq(X,m[:,:],rcond=None)[0]; print(name,'y and final state',list(zip(p.get_feature_names_out(),W[:,[0,-1]].round(4).tolist())))
 for j in [0,-1]:
  C=np.zeros((4,4,4));Q=np.zeros((4,4));
  for inds in np.ndindex(C.shape):
   a=np.bincount(inds,minlength=4); C[inds]=W[np.all(p.powers_==a,axis=1),j].item()*np.prod([factorial(k) for k in a])/6
  for inds in np.ndindex(Q.shape):
   a=np.bincount(inds,minlength=4); Q[inds]=W[np.all(p.powers_==a,axis=1),j].item()*np.prod([factorial(k) for k in a])/2
  print('j',j,'Q eigen',np.linalg.eigh(Q),'C sv',np.linalg.svd(C.reshape(4,-1),full_matrices=False)[1]);print('C modes',np.linalg.svd(C.reshape(4,-1),full_matrices=False)[0])
p=PolynomialFeatures(6); P=p.fit_transform(x); d=p.powers_.sum(axis=1); n=52
res=[]
for weighting in ['monomial','isotropic']:
 for cr in [.25,.5,1]:
  for fac in [0,.075,.15,.3,.5]:
   S=cr**np.maximum(d-2,0).clip(max=1)*fac**np.maximum(d-3,0)
   if weighting=='isotropic': S=S*np.sqrt(np.array([factorial(a.sum())/np.prod([factorial(k) for k in a]) for a in p.powers_]))
   H=(P*S)@(P*S).T
   for a in [1e-5,.0002,.001,.004,.012,.035]:
    inv=np.linalg.inv(H[:n,:n]+a*np.eye(n)); C=inv@m[:n]; E=abs(C/np.diag(inv)[:,None]); R=abs(H[n:,:n]@C-m[n:]); res.append((E[:,0].mean(),E[:,1:].mean(),R[:,0].mean(),R[:,1:].mean(),weighting,cr,fac,a))
for j in [0,1]: print('top LOO target',j,*[tuple(np.round(r[:4],5))+r[4:] for r in sorted(res,key=lambda r:r[j])[:8]],sep='\n')
from copy import deepcopy
for name,idx in [('prescribed',np.arange(52,72)),('large norm',np.argsort(np.linalg.norm(x,axis=1))[-20:]),('large max coordinate',np.argsort(abs(x).max(1))[-20:]),('first 20',np.arange(20))]+[(f'random{i}',np.random.default_rng(i).choice(72,20,False)) for i in range(5)]:
 tr=np.setdiff1d(np.arange(72),idx); S=deepcopy(s); S['train']={'x':x[tr],'y':y[tr],'traj':z[tr]}; S['test']={'x':x[idx]}; o=solve_batch(S); e=abs(o['y_hat']-y[idx]); f=abs(o['state_pred']-z[idx]); print(name, np.round([e.mean(),e.max(),f.mean(),f.mean(0).max(),f.max()],5))
