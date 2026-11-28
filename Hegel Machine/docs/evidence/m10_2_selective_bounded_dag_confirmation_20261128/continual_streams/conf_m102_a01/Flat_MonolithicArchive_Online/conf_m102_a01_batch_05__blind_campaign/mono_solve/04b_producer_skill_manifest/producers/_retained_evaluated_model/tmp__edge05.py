import sys,pickle
import numpy as np
from scipy.linalg import cho_factor,cho_solve
sys.path.insert(0,'/holdout_task')
from batch_solver import _coordinates,_fixed,_adaptive
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; n=len(x); y=t['y']; traj=t['traj']; z=_coordinates(s['panel_a'],x)
b=s['panel_b']; cb=np.linalg.lstsq(np.c_[np.ones(len(b['u'])),b['u']],b['v'],rcond=None)[0]; h=np.linalg.solve(cb[1:].T,(z-cb[0]).T).T
rng=np.random.default_rng(47219); partitions=[('ordered',np.arange(n))]+[('random',rng.permutation(n)) for _ in range(16)]
for j in range(4):
 ix=np.argsort(x[:,j]); partitions.extend([('edge',ix),('edge',ix[::-1])])
m={}
for tag,ix in partitions:
 i,j=ix[:52],ix[52:]; _,_,vt=np.linalg.svd(traj[i],full_matrices=False); mode=vt[0]; a=traj[i]@mode; Y=np.c_[y[i],a]
 def score(name,p):
  err=np.r_[np.mean(abs(p[:,0]-y[j])),np.mean(abs(p[:,1,None]*mode-traj[j])),np.mean(abs(p[:,1]*mode[-1]-traj[j,-1]))]; m.setdefault(name,{}).setdefault(tag,[]).append(err)
 w0=z[ix]; w0=(w0-w0[:52].mean(0))/w0[:52].std(0); base=np.c_[.75*_fixed(w0,y[i],2.2)+.25*_adaptive(w0,y[i]),.7*_adaptive(w0,a)+.3*_fixed(w0,a,5.)]; score(('baseline',),base)
 for q,w in [('z',z[ix]),('h',h[ix])]:
  w=w/max(np.sqrt(np.mean(w[:52]**2)),.1); G=w@w[:52].T/4; eye=np.eye(52); diff=w[:,None]-w[None,:52]; dd=(diff**2).sum(2); pol=1+G+G**2
  for d in [3,4,5,6,8]:
   for rho in [.04,.12,.35,.8]:
    K=pol+sum(rho**(p-2)*G**p for p in range(3,d+1))
    for noise in [1e-5,.0002,.002]:
     p=K[52:]@cho_solve(cho_factor(K[:52]+noise*eye),Y); name=(q,'poly',d,rho,noise); score(name,p);score(name+('mix',),.5*(p+base))
  for length in [1.3,2.,3.,4.5]:
   R=np.exp(-.5*dd/length**2); U=np.exp(-.5*diff**2/length**2); additive=U.sum(2); pair=sum(U[:,:,k]*U[:,:,l] for k in range(4) for l in range(k+1,4));
   for kind,r in [('rbf',R),('add',additive),('pair',pair)]:
    for weight in [.03,.3,3.]:
     for noise in [1e-5,.0002,.002]:
      K=pol+weight*r; p=K[52:]@cho_solve(cho_factor(K[:52]+noise*eye),Y); name=(q,kind,length,weight,noise);score(name,p);score(name+('mix',),.5*(p+base))
def summary(a):return np.array([a['ordered'][0],np.mean(a['random'],0),np.max(a['random'],0),np.mean(a['edge'],0),np.max(a['edge'],0)])
print('baseline',summary(m[('baseline',)]).round(5));
for target in range(2):
 v=[(k,summary(a)) for k,a in m.items()]; v=[(k,a) for k,a in v if a[0,target] < (.039 if target==0 else .018) and a[1,target] < (.026 if target==0 else .016)]; print('target',target);
 for k,a in sorted(v,key=lambda q: q[1][:,target]@np.array([.5,1.,.35,.6,.35]))[:9]: print(k,a[:,target].round(5), 'terminal',a[:,2].round(5))
print('train/test edge support',np.min(x,0),np.max(x,0),np.min(s['test']['x'],0),np.max(s['test']['x'],0))
