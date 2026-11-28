import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['panel_c']; x=c['u']; t=np.c_[c['v1'],c['v2']]
D={}
for i in range(4): D[str(i)]=x[:,i]
for k in (2,3):
 for p in combinations_with_replacement(range(4),k): D['*'.join(map(str,p))]=np.prod(x[:,p],axis=1)
for i in range(4):
 for j in range(i+1,4):
  for a in [-2,-1,-.5,.5,1,2]: D[f'{i}+{a:g}*{j}']=x[:,i]+a*x[:,j]
A=np.c_[np.ones(len(x)),x]; B=np.c_[np.ones(len(x)),*[D[k] for k in list(D)[:14]]]
def score(M,y):
 q=np.linalg.pinv(M); w=q@y; r=y-M@w; h=np.sum(M*q.T,axis=1); return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/np.maximum(1-h,1e-12))**2)),w
print('ranges x',np.round(np.stack([x.min(0),x.max(0)]),3)); print('ranges targets',np.round(np.stack([t.min(0),t.max(0)]),3))
F={k:v for k,v in D.items() if '+' not in k}
for k,z in D.items():
 for a in [.5,1,1.5,2,3]:
  for f,g in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda z:z/(1+z*z)),('abs',np.abs)]: F[f'{f}({a:g}*{k})']=g(a*z)
for k,z in list(D.items())[:14]:
 for j,w in D.items():
  if '*' in j and '+' not in j: continue
  for a in [.3,1.]: F[f'{k}/(1+{a}*({j})^2)']=z/(1+a*w*w)
for j,y in enumerate(t.T):
 print('target',j,'lin',score(A,y)[:2], 'quad',score(B,y)[:2], 'lincoef',np.round(score(A,y)[2],3))
 for name,base in [('constant',A[:,:1]),('linear',A)]:
  out=sorted((score(np.c_[base,v],y)[1],k) for k,v in F.items()); print(name,[(round(a,4),k) for a,k in out[:6]])
 m=A.copy(); sel=[]
 for _ in range(4):
  _,k=min((score(np.c_[m,v],y)[1],k) for k,v in F.items() if k not in sel); sel.append(k); m=np.c_[m,F[k]]
 print('greedy',sel,np.round(score(m,y)[0:2],4),np.round(score(m,y)[2],3))
 for f in [np.arctanh,np.arcsinh]:
  if f==np.arctanh and np.max(abs(y))>=1: continue
  print('inverse',f.__name__,*[np.round(score(M,f(y))[0:2],4) for M in (A,B)])
