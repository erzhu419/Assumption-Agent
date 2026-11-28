from probe import *
from scipy.optimize import least_squares
from scipy.special import expit

def calibration(s):
 p=s['panel_a']; u,v=p['u'],p['v']; keep=np.linalg.norm(v,axis=1)<5
 def features(u,j):
  a=u[:,j]; b=u[:,(j+1)%4]; return np.c_[pol(u,1),a*b,a**3]
 cc=np.column_stack([fit(features(u[keep],j),v[keep,j]) for j in range(4)])
 def f(u):return np.column_stack([features(u,j)@cc[:,j] for j in range(4)])
 def inv(x):
  guess=np.linalg.solve(cc[1:5].T,(x-cc[0]).T).T
  ret=[]
  for a,b in zip(guess,x):
   if np.linalg.norm(f(a[None])[0]-b)<1e-12:ret.append(a); continue
   fn=lambda z:f(z[None])[0]-b
   r=least_squares(fn,a,max_nfev=80,gtol=1e-12); pool=[r.x]
   if np.linalg.norm(r.fun)>1e-7 or np.max(abs(r.x))>1.35:
    for ini in [np.clip(a,-1.2,1.2),np.zeros(4), *[a+k for k in [np.eye(4)[i]*v for i in range(4) for v in [-1,1]]]]:
     pool.append(least_squares(fn,ini,max_nfev=80,gtol=1e-12).x)
   ret.append(min(pool,key=lambda z: 1e4*np.linalg.norm(fn(z)) + np.sum(abs(z)**4)))
  return np.array(ret)
 return inv, f, cc, np.std(f(u[keep])-v[keep],axis=0)

def operators(s):
 ops={}
 for pn,kn in [('b','v'),('c','v1'),('c','v2')]:
  u,v=[s['panel_'+pn][k] for k in ['u',kn]]
  candidates=[]
  for name,fn in [('lin+tanh',lambda u,h:np.c_[pol(u,1),np.tanh(h[0]*u)]),('lin+sin',lambda u,h:np.c_[pol(u,1),np.sin(h[0]*u)]),('lin+cos',lambda u,h:np.c_[pol(u,1),np.cos(h[0]*u)]),('tanh+cos',lambda u,h:np.c_[np.ones(len(u)),np.tanh(h[0]*u),np.cos(h[1]*u)]),('tanh+sin',lambda u,h:np.c_[np.ones(len(u)),np.tanh(h[0]*u),np.sin(h[1]*u)]),('sin+cos',lambda u,h:np.c_[np.ones(len(u)),np.sin(h[0]*u),np.cos(h[1]*u)]),('lin+tanh+cos',lambda u,h:np.c_[pol(u,1),np.tanh(h[0]*u),np.cos(h[1]*u)]),('lin+tanh+sin',lambda u,h:np.c_[pol(u,1),np.tanh(h[0]*u),np.sin(h[1]*u)]),('lin+sin+cos',lambda u,h:np.c_[pol(u,1),np.sin(h[0]*u),np.cos(h[1]*u)])]:
   def err(h):
    a=fn(u,h);return (a@fit(a,v)-v).ravel()
   sol=min((least_squares(err,[a,b],bounds=(.2,6),max_nfev=80) for a,b in [(1,1),(2,2),(3,1.25),(1.25,3),(4,4)]),key=lambda r:r.cost); h=sol.x; a=fn(u,h); c=fit(a,v); rms=np.std(err(h)); candidates.append((rms,name,h,fn,c));
  candidates.sort(key=lambda v:v[0]); print('operator',kn,[(r[1],np.round(r[0],6),np.round(r[2],4)) for r in candidates]);
  rms,name,h,fn,c=min(candidates,key=lambda r:r[0]+.00005*len(r[4])); print('selected',name,h,c.T); ops[kn]=lambda x,fn=fn,h=h,c=c:fn(x,h)@c
 return ops

if __name__=='__main__':
 inv,f,cc,err=calibration(s); print('A',cc.T,err)
 x,y,t=[s['train'][k] for k in ['x','y','traj']]; z=inv(x); print('calibration inversion errors, range',np.sort(np.linalg.norm(f(z)-x,axis=1))[-10:],z.min(0),z.max(0)); o=operators(s); obs=np.c_[y,t[:,[0,-1]]]; results=[]
 Z={'x':x,'a':z};
 for a in list(Z):
  for b in o: Z[a+b]=o[b](Z[a]);
 for a in list(Z)[2:]:
  for b in o: Z[a+b]=o[b](Z[a]);
 for k,w in Z.items():
  for d in (1,2):
   A=pol(w,d); c=fit(A[:52],obs[:52]); e=np.mean(abs(obs-A@c)[52:],axis=0); results.append((e[2],k,d,e,np.std(obs-A@c,axis=0)))
 for r in sorted(results,key=lambda r:r[0])[:25]:print(r)
 print('best y');
 for r in sorted(results,key=lambda r:r[3][0])[:12]:print(r)
