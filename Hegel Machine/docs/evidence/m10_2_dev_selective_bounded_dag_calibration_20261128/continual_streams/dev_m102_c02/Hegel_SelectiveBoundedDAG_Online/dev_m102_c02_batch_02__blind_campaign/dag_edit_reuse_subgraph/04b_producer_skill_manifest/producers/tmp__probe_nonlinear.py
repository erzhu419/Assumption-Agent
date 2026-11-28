import itertools as it, pickle, runpy
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['c']; y=s['train']['y']; n=len(y)
a,b,d,e=c.T; p=a*d+b*e; base=np.c_[np.ones(n),c,p]; tr=np.arange(52); va=np.arange(52,n)
def resid(X,idx=tr):
 return y-X@np.linalg.lstsq(X[idx],y[idx],rcond=None)[0]
r=resid(base); print('res quantiles',np.quantile(r[tr],[0,.05,.25,.5,.75,.95,1])); print('largest residuals',np.c_[c[tr],y[tr],r[tr]][np.argsort(abs(r[tr]))[-10:]].round(3)); print('corr residual squared with coord',np.corrcoef(np.c_[c,abs(r)].T)[-1,:4])
F={}; C={str(i):v for i,v in enumerate(c.T)}; P={''.join(map(str,ix)):np.prod(c[:,ix],axis=1) for k in (2,3,4) for ix in it.combinations_with_replacement(range(4),k)}
F.update(P)
L={**C,'p':p,'det':a*e-b*d,'cross':a*e+b*d,'diff':a*d-b*e,'sq':np.sum(c*c,axis=1),'sq01':a*a+b*b,'sq23':d*d+e*e}
for i,j in it.combinations(range(4),2):
 for k in (-2,-1,-.5,.5,1,2):L[f'{i}+{k}*{j}']=c[:,i]+k*c[:,j]
for name,z in L.items():
 for k in (.25,.5,1,1.25,2,3,4):
  for fn,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('abs',np.abs),('softclip',lambda t:t/(1+t*t)),('sat',lambda t:t/np.sqrt(1+t*t))]:F[f'{fn}({k}*{name})']=f(k*z)
 for nm,q in {**C,**{i:j for i,j in P.items() if len(i)==2},'p':p,'det':L['det'],'1':np.ones(n)}.items():
  for k in (.25,.5,1,1.25,2,4): F[f'{nm}/(1+{k}*{name}^2)']=q/(1+k*z*z)
  F[f'{nm}*abs({name})']=q*abs(z)
D=np.column_stack(list(F.values())); names=np.array(list(F)); rng=np.random.default_rng(119); folds=np.array_split(rng.permutation(52),5)
for label,B in [('archive',base),('affine',base[:,:5])]:
 print(label)
 for step in range(4):
  Q=np.linalg.qr(B[tr])[0]; Z=D[tr]-Q@(Q.T@D[tr]); norm=np.linalg.norm(Z,axis=0); rr=resid(B)[tr]; score=abs(rr@Z)/np.maximum(norm,1e-8); order=np.argsort(-score)[:12]; metrics=[]
  for i in order:
   X=np.c_[B,D[:,i]]; R=resid(X); fm=[np.mean(abs(resid(X,np.setdiff1d(tr,j))[j])) for j in folds]; metrics.append((names[i],round(np.mean(abs(R[tr])),5),round(np.mean(abs(R[va])),5),round(max(fm),5)))
  print(step,metrics[:8]); B=np.c_[B,D[:,order[0]]]
for k in (2,3,4,5,6):
 X=np.c_[base[:,:5],*[np.prod(c[:,ix],axis=1) for j in range(2,k+1) for ix in it.combinations_with_replacement(range(4),j)]]; R=resid(X); print('polynomial',k,X.shape[1],np.mean(abs(R[tr])),np.mean(abs(R[va])))
