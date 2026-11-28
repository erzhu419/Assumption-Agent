import sys
sys.path.insert(0,'/holdout_task')
import batch_solver as m
import numpy as np,pickle
from itertools import combinations, combinations_with_replacement
from scipy.linalg import solve
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];y=s['train']['y']; t=s['train']['traj']; ids=np.arange(len(x))
A=m._calibration(s['panel_a']);z=m._decode(x,A); ops=m._panel_operators(s['panel_c']);a,b=(f(z) for f in ops);B=m._fit(m._affine(s['panel_b']['u']),s['panel_b']['v']);w=m._affine(z)@B
folds=[(ids[:52],ids[52:])];
for seed in [29,487,8701]:
 perm=np.random.default_rng(seed).permutation(len(x));folds += [(np.setdiff1d(ids,j),j) for j in np.array_split(perm,4)]
D=lambda X:m._affine(X)
H=m._scalar_design(z,ops)
for design,label in [(z,'traj: decoded affine'),(w,'traj: transport relaxed')]:
 es=[]
 for i,j in folds:
  yh=D(design[j])@m._fit(D(design[i]),t[i]) if design is z else design[j]@m._relaxation(design[i],t[i],s['eval_times']);es.append(np.mean(abs(yh-t[j]),axis=0))
 print(label,np.round(es[0],5),'worst per horizon',np.max(es,axis=0).round(5))

def poly(v,d):return np.column_stack([np.prod(v[:,ix],axis=1) for ix in combinations_with_replacement(range(v.shape[1]),d)])
F={}
F['base']=H
F['base+xyz']=np.c_[H,poly(z,2)]
F['base+xyz3']=np.c_[H,poly(z,2),poly(z,3)]
F['base+z_sep']=np.c_[H,z**2,z**3,np.tanh(1.25*z),np.sin(1.25*z),np.cos(1.25*z)]
F['base+z_sep+xyz']=np.c_[F['base+z_sep'],poly(z,2)]
F['base+allpanel']=np.c_[H,poly(np.c_[a,b],2)]
F['sep_w']=np.c_[z,a,b,poly(w,2),z**3,np.tanh(1.25*z),z/(1+.22*z*z)]
F['z3']=np.c_[z,poly(z,2),poly(z,3)]
F['z_sep+quad']=np.c_[z,poly(z,2),z**3,np.tanh(1.25*z),np.sin(1.25*z),np.cos(1.25*z)]
F['separable with pairwise']=np.c_[z,poly(z,2),z**3, z/(1+.22*z*z),z**2/(1+.22*z*z),*[z[:,i,None] * np.c_[z[:,j]**2,np.tanh(1.25*z[:,j]),np.cos(1.25*z[:,j])] for i in range(4) for j in range(4) if i!=j]]

def err(p, j): return np.mean(abs(p-y[j]))
def report(label, errors):return (np.mean(errors[1:]),np.max(errors),errors[0],label)
results=[]
for label,h in F.items():
 for lam in [.001,.01,.05,.2,1.]:
  es=[err(m._ridge_predict(h[i],h[j],y[i],lam),j) for i,j in folds]; results.append(report((label,lam),es))
for dim in ['z','w','za','zab']:
 h={'z':z,'w':w,'za':np.c_[z,a],'zab':np.c_[z,a,b]}[dim]
 for kind in ['rbf','matern52','add','add+pair']:
  for l in [.4,.7,1.1,1.8,3.]:
   for lam in [1e-5,1e-3,.03,.3]:
    for prior in ['z','H']:
     es=[]
     for i,j in folds:
      v=h/h[i].std(0);d=cdist(v,v)/l
      d1=abs(v[:,None,:]-v[None,:,:])/l
      t0=np.exp(-.5*d1**2)
      if kind=='rbf':K=np.exp(-.5*d*d)
      elif kind=='matern52':q=np.sqrt(5)*d;K=(1+q+q*q/3)*np.exp(-q)
      elif kind=='add':K=t0.mean(2)
      else:K=t0.mean(2)+np.mean([t0[:,:,k]*t0[:,:,n] for k,n in combinations(range(v.shape[1]),2)],axis=0)
      d0=H if prior=='H' else z; d0=(d0-d0[i].mean(0))/d0[i].std(0);K += (d0@d0.T)*(.25 if prior=='H' else 1.)+1.
      pred=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),y[i],assume_a='pos');es.append(err(pred,j))
     results.append(report(('kernel',dim,kind,l,lam,prior),es))
print('by mean',*sorted(results)[:22],sep='\n');print('by worst',*sorted(results,key=lambda q:q[1])[:16],sep='\n');print('base',*[r for r in results if r[-1][0]=='base'],sep='\n')
