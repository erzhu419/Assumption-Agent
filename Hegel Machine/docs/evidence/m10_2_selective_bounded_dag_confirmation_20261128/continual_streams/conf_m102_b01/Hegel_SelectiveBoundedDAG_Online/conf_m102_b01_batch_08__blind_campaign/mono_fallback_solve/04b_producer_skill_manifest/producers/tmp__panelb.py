from base import *
from itertools import combinations
pairs=list(combinations(range(4),2));
def feat(x,q,kind):
 p=q[:2]; w=x+p[0]*np.sin(p[1]*x); ww=w
 if kind=='per-coordinate':w=x+q[:4]*np.sin(q[4:8]*x)
 if kind=='per-cbrt':w=x+q[:4]*np.sin(q[4:8]*x)+q[8:12]*np.cbrt(x)
 if kind=='per-cubic':w=x+q[:4]*np.sin(q[4:8]*x)+q[8:12]*x**3
 if kind=='per-q':w=x+q[:4]*np.sin(q[4:8]*x)+q[8:12]*x**2
 if kind=='cbrt':w=w+q[2]*np.cbrt(x)
 if kind=='cubic':w=w+q[2]*x**3
 if kind=='cross':w=np.c_[w,*[x[:,i]*x[:,j] for i,j in pairs]]
 if kind=='crossw':w=np.c_[w,*[ww[:,i]*ww[:,j] for i,j in pairs]]
 if kind=='crosstrig':w=np.c_[w,*[np.sin(q[2]*x[:,i]*x[:,j]) for i,j in pairs]]
 if kind=='ext':w=np.c_[w,x*x]
 if kind=='rat':w=x/(1+q[2]*x*x)+q[0]*np.sin(q[1]*x)
 if kind=='cubrt':w=np.cbrt(x)+q[0]*x+q[1]*x**3
 return aff(w)
def fit(kind,out,q0):
 def res(q,idx=slice(None)):
  f=feat(u[idx],q,kind); g=v[idx]; k=q[-1]
  if out=='tanh': h=np.arctanh(np.clip(k*g,-.999,.999))/k; back=lambda a:np.tanh(k*a)/k
  elif out=='asinh': h=np.sinh(k*g)/k;back=lambda a:np.arcsinh(k*a)/k
  elif out=='sinh': h=np.arcsinh(k*g)/k;back=lambda a:np.sinh(np.clip(k*a,-10,10))/k
  else: h=g; back=lambda a:a
  co=ls(f,h); return (back(f@co)-g).ravel(),co,back
 q=least_squares(lambda q:res(q)[0],q0,max_nfev=150).x; r,co,back=res(q); a=feat(u,q,kind); h=(a*np.linalg.pinv(a).T).sum(1); val=np.std(r.reshape(-1,4)/(1-h[:,None]));print(kind,out,'r',round(np.std(r),7),'loo*',round(val,7),np.round(q,5));return q,co,back,val
results={}
for name,q0 in {'sin':[-.4,2.9],'cbrt':[-.4,2.9,0.02],'cubic':[-.4,3.8,1.2],'rat':[-.4,2.9,.01],'per-coordinate':[-.4]*4+[2.9]*4,'per-cubic':[-.4]*4+[2.9]*4+[.01]*4,'per-cbrt':[-.4]*4+[2.9]*4+[.02]*4,'per-q':[-.4]*4+[2.9]*4+[.01]*4,'cross':[-.4,2.9],'crossw':[-.4,2.9],'crosstrig':[-.4,2.9,1.],'ext':[-.4,2.9]}.items():
 for out in ('identity','tanh','asinh','sinh') if name in ('sin','cbrt','cubic','rat','per-coordinate') else ('identity',):
  q,co,back,val=fit(name,out,q0+([.15] if out!='identity' else []));results[name+'_'+out]=(back(feat(z,q,name)@co),val)
np.savez('/tmp/Bs.npz',**{k:b for k,(b,vv) in results.items()});
for key in sorted(results,key=lambda k:results[k][1])[:8]:
 B=results[key][0]; C=m._pairmap(B)@C0; c=np.arcsinh(1.25*C);d=np.tanh(1.25*D);print('readout',key,score(np.c_[c,d,(c[:,:,None]*d[:,None,:]).reshape(-1,16)],.25).round(4).tolist())
T=Y[:,1:]; _,_,vt=np.linalg.svd(T-T.mean(0),full_matrices=False); Tm=T.mean(0); print('mean time',Tm,'ratio',vt[0]/vt[0,-1]);tt=s['eval_times'];
for name,func,q0 in [('exp',lambda q:1-np.exp(-q[0]*tt),[.7]),('log',lambda q:np.log1p(q[0]*tt),[2.]),('rat',lambda q:tt/(1+q[0]*tt),[.4]),('pow',lambda q:tt**q[0],[.6]),('lxp',lambda q:np.log1p(q[0]*tt)**q[1],[2,1]),('dual',lambda q:1-np.exp(-q[0]*tt),[.7])]:
 def obj(q):
  a=func(q);a=a/a[-1]; return a-vt[0]/vt[0,-1]
 q=least_squares(obj,q0,bounds=(.001,20)).x;print(name,q,np.linalg.norm(obj(q)),obj(q).round(4))
