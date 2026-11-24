import numpy as np
from graph_study import data,base
from profile_graph import profile,fit

def trial(C,n):
 F=base(C,.06,'raw'); G=base(C,0,'cov'); p=len(C); i,j=np.triu_indices(p,1)
 for name,U in [('prec',F),('cov',G)]:
  S=fit(C,U,.019); f,_,R=profile(C,S); T=np.linalg.inv(R)
  obj=f+2*.019*np.sum(np.hypot(S[i,j],.012)-.012)
  yield name,obj,T

def stats(T,truth,n=900):
 i,j=np.triu_indices(len(T),1); y=truth[i,j]; v=np.abs(T[i,j])/np.sqrt(T[i,i]*T[j,j]); order=np.argsort(v)[::-1];
 return [2*y[v>t].sum()/(y.sum()+(v>t).sum()) for t in (.102,.112,.122,.132)] + [2*y[order[:k]].sum()/(y.sum()+k) for k in (26,28,30)]

for mode in ('publiclike','uneven','weakfactor','mixed','random'):
 out={}
 for seed in range(35):
  X,a,_=data(seed,mode); C=np.cov(X,rowvar=False); (p,op,P),(c,oc,Q)=trial(C,len(X));
  for k,T in [('prec',P),('cov',Q),('best',P if op<oc else Q),('avg',(P+Q)/2)]:out.setdefault(k,[]).append(stats(T,a))
 print(mode)
 for k,x in out.items(): print(k,np.round(np.mean(x,axis=0),3),np.round(np.min(x,axis=0),3))
