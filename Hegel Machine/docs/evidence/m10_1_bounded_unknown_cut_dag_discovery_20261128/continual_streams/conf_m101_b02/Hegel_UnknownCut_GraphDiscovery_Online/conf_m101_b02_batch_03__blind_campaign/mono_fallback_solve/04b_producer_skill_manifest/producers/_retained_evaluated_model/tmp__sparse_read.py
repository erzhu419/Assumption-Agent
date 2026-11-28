import numpy as np
from scipy.linalg import solve
from sklearn.linear_model import ARDRegression, lars_path
from sklearn.model_selection import KFold
z=np.load('/tmp/latent.npz'); f,c,d,o,e=[z[k] for k in ('f','c','d','o','e')]; T=z['T']; t=z['times']; h=-np.expm1(-t[:,None]*[1.82,.417]); Y=np.c_[T[:,0], np.linalg.lstsq(h,T[:,1:].T,rcond=None)[0].T]; x=z['x']
folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=123).split(x)]
def dictionary(blocks, degree):
 import itertools
 xx=np.concatenate(blocks,axis=1); n=xx.shape[1]; rows=[]; names=[]
 for k in range(1,degree+1):
  for j in itertools.combinations_with_replacement(range(n),k):
   rows.append(np.prod(xx[:,j],axis=1)); names.append(j)
 return np.column_stack(rows),names
A={}
for key,blocks in [('mixed',[f,c,d]),('native',[f,o,e]),('rawmixed',[x,c,d])]:
 A[key]=dictionary(blocks,2)
for k in [-1,0,1,2]:
 blocks=[f,c,d]; A[f'aligned{k}']=(np.column_stack([*blocks,*(a*np.roll(b,k,axis=1) for a in blocks for b in blocks)]),[])
A['elementary']=(np.c_[f,o,e,f*o,f*e,o*e,f**2,o**2,e**2],[])
A['mixed_cubic']=dictionary([f,c,d],3)
A['broad']=(np.c_[A['mixed'][0],np.sin(f),np.cos(f),np.sin(c),np.cos(c),np.sin(d),np.cos(d)],[])
def regress(g,i,j,method,v):
 a=g[i]; mu=a.mean(0); std=np.maximum(a.std(0),1e-10); a=(a-mu)/std; b=(g[j]-mu)/std; yy=Y[i]; ymu=yy.mean(0); yy=yy-ymu
 if method=='ridge':
  yp=b@a.T@solve(a@a.T+v*len(a)*np.eye(len(a)),yy,assume_a='pos')
 elif method=='ard':
  yp=np.column_stack([ARDRegression(threshold_lambda=v).fit(a,q).predict(b) for q in yy.T])
 else:
  out=[]
  for q in yy.T:
   _,_,w=lars_path(a,q,method='lasso',max_iter=v); active=np.flatnonzero(abs(w[:,-1])>1e-10); mat=a[:,active]; out.append(b[:,active]@np.linalg.lstsq(mat,q,rcond=None)[0])
  yp=np.column_stack(out)
 yp+=ymu; return np.c_[yp[:,0],yp[:,1:]@h.T]
res=[]
for key,(g,names) in A.items():
 for method,vals in [('ridge',(1e-5,1e-4,3e-4,.001,.003,.01)),('ard',(100,1000,10000)),('lars',(5,10,15,20,30))]:
  if key=='mixed_cubic' and method=='ard':continue
  for v in vals:
   es=[]
   for i,j in folds:
    delta=abs(regress(g,i,j,method,v)-T[j]);es.append([delta[:,0].mean(),delta[:,1:].mean()])
   es=np.array(es);res.append((key,method,v,*np.round(es[0],5),*np.round(es[1],5),*np.round(es[2:].mean(0),5),*np.round(es[2:].max(0),5)))
for k in (3,4,7,8,9,10):
 print('metric',k);print(*sorted(res,key=lambda row:row[k])[:6],sep='\n')
np.savez('/tmp/approaches.npz',**{k:g for k,(g,_) in A.items()})
