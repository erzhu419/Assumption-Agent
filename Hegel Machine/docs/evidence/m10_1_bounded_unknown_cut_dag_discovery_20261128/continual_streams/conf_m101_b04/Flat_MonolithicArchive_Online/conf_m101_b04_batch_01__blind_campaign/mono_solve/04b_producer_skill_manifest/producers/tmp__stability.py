import sys
sys.path[:0]=['/holdout_task','/tmp']
from batch_solver import _bank,_coordinates,_quadratic,_fit,_selected
from dagprobe import s,blocks,poly,cdist
from scipy.linalg import solve
import numpy as np
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; n=len(x); ids=np.arange(n); q=_quadratic(x); a,b=_coordinates(s,x); f=_bank(x,a,b); d=blocks(x,s)
splits=[(ids[:52],ids[52:]),(ids[:56],ids[56:])]+[(np.setdiff1d(ids,j),j) for j in np.array_split(np.random.default_rng(13).permutation(n),4)]; out={}
def store(k,p,j):out.setdefault(k,[]).append((p,y[j]))
K0=np.exp(-.23*cdist(x,x,'sqeuclidean'))
for i,j in splits:
 R=np.c_[y,t[:,-3:].mean(1)]; rng=np.random.default_rng(62); bag={}
 for repeat in range(64):
  idx=i if repeat==0 else rng.choice(i,int(.83*len(i)),replace=False)
  order=_selected(q[idx],f[idx],R[idx],12)
  for k in (4,6,8,10,12):
   F=np.c_[q,f[:,order[:k]]]; bag.setdefault(k,[]).append(F[j]@_fit(F[i],y[i]));
 for k,P in bag.items():
  store(('bag',k),np.mean(P[1:],0),j); store(('median',k),np.median(P[1:],0),j); store(('omp',k),P[0],j)
 N=np.linalg.qr(q[i],mode='complete')[0][:,15:]; P=N.T@f[i]; norms=np.sqrt(np.mean(P*P,0)); mask=norms>1e-7
 def kernel(F,unit=False):
  v=N.T@F[i]; scale=(np.sqrt(np.mean(v*v,0))+1e-8) if unit else np.sqrt(np.mean(v*v))+1e-8; Z=F/scale;return Z@Z.T/Z.shape[1]
 kernels={'raw':K0,'bank':kernel(f[:,mask]),'banku':kernel(f[:,mask],True)}
 valid=['a','b','ca','da','cb','db'];
 for k in valid:
  z=d[k]; kernels[k+'p']=kernel(poly(z,3)[:,5:]); kernels[k+'l']=kernel(z); kernels[k+'r']=np.exp(-.23*cdist(z,z,'sqeuclidean'))
 for name,keys in [('poly',[k+'p' for k in valid]),('pan',[k+'l' for k in valid[2:]]),('kr',[k+'r' for k in valid]),('cross',['cbp','dbp','cap','dap'])]:kernels[name]=np.mean([kernels[k] for k in keys],0)
 for name,K in kernels.items():
  if not np.all(np.isfinite(K)) or K.max()>1e14:continue
  for g in (0,.25,1):
   for p in (.001,.01,.05,.2):
    k=K+g*K0; v=k[np.ix_(i,i)]; c=N@solve(N.T@v@N+p*np.eye(N.shape[1]),N.T@y[i],assume_a='pos'); pred=q[j]@_fit(q[i],y[i]-v@c)+k[np.ix_(j,i)]@c;store((name,g,p),pred,j)
M={k:np.array([np.mean(abs(p-yj)) for p,yj in v]) for k,v in out.items()}
for group in ['gate','worst','avg']:
 print(group)
 for k in sorted(M,key=lambda k: M[k][0] if group=='gate' else max(M[k]) if group=='worst' else np.mean(M[k]))[:10]:print(k,M[k].round(5))
print('omp bag');print([(k,np.round(v,4)) for k,v in M.items() if k[0] in ('omp','bag','median')]);
for base in [('omp',10),('bag',10),('bag',6)]:
 rows=[]
 for k,v in out.items():
  for w in (.25,.5,.75):
   e=np.array([np.mean(abs(w*p+(1-w)*b-yj)) for (p,yj),(b,_) in zip(v,out[base])]);rows.append((e,k,w))
 print('blend',base)
 for e,k,w in sorted(rows,key=lambda r:max(r[0])+.5*r[0][1])[:16]:print(k,w,e.round(5))
 for e,k,w in sorted((r for r in rows if r[0][0]<=.038),key=lambda r:max(r[0])+.5*r[0][1])[:8]:print('passes',k,w,e.round(5))
