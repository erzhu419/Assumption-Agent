exec(open('/tmp/probe.py').read().split('ops={}')[0]); from scipy.optimize import least_squares
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from itertools import product
B=fit(ext(b['u']),b['v']); n=52
u=c['u']; v=c['v1']
for mode in ('tanh','tanhout','tanhlin','sin','softsign'):
 def f(z,k):
  if mode=='tanh': return ext(np.tanh(z*k))
  if mode=='tanhlin':return ext(np.c_[z,np.tanh(z*k)])
  if mode=='sin':return ext(np.sin(z*k))
  if mode=='softsign':return ext(z/(1+k*abs(z)))
  return ext(z)
 def inv(z,k): return np.arctanh(np.clip(z*k,-.99999,.99999))/k if mode=='tanhout' else z
 def e(k):
  vv=inv(v,k); F=f(u,k); return (F@fit(F,vv)-vv).ravel()
 k=least_squares(e,(.5/abs(v).max(axis=0) if mode=='tanhout' else np.repeat(1.5,4)),bounds=(.04,.99/abs(v).max(axis=0) if mode=='tanhout' else 5)).x; print('c1',mode,k,np.sqrt(np.mean(e(k)**2)))
 if mode=='tanh': C1=fit(f(u,k),v); k1=k
k2=minimize_scalar(lambda k: np.mean((ext(np.c_[u,u*u/(1+k*u*u)])@fit(ext(np.c_[u,u*u/(1+k*u*u)]),c['v2'])-c['v2'])**2),bounds=(.05,.5),method='bounded').x
C2=fit(ext(np.c_[u,u*u/(1+k2*u*u)]),c['v2']); print('c2',k2,C2.round(3)); print('t',t)
mode=np.linalg.svd(T[:n],full_matrices=False)[2][0]; mode*=np.sign(mode[-1]); P=T@mode; Y=np.c_[y,P]; print('svd',mode,'yt corr',np.corrcoef(Y.T)[0,1]);
z=(x-cal[1])/cal[0]; zs={}
for label,w in [('z',z),('x',x),('B',(z-B[0])@np.linalg.inv(B[1:])),('b',ext(z)@B)]:
 d=ext(np.tanh(w*k1))@C1; e=ext(np.c_[w,w*w/(1+k2*w*w)])@C2
 zs.update({label:w,label+'c1':d,label+'c2':e,label+'both':np.c_[d,e],label+'all':np.c_[w,d,e]})
res=[]
for key,w in zs.items():
 for Fkey,F in [('lin',w),('quad',np.c_[w,quad(w)]),('joint',np.c_[w,quad(w),w**3,w**4])]:
  F=ext(F); F/=np.maximum(np.std(F[:n],axis=0),.1); K=F@F.T
  for l in (1e-6,.003,.03,.2,1.):
   A=cho_solve(cho_factor(K[:n,:n]+l*np.eye(n)),np.eye(n)); beta=A@Y[:n]; e=beta/np.diag(A)[:,None]; q=K[n:,:n]@beta; vac=np.mean(abs(q-Y[n:]),axis=0); tr=np.mean(abs(e),axis=0); res.append((*tr,*vac,key,Fkey,l))
for j in range(4):print('ranking',j,*sorted(res,key=lambda r:r[j])[:6],sep='\n')
