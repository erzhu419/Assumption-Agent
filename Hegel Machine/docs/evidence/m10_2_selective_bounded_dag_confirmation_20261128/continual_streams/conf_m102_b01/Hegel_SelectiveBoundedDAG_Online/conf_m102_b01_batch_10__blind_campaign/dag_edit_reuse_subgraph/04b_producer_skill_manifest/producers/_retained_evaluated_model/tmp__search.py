from maps import *
from scipy.optimize import least_squares

def inverse(f,coef,x):
 zz=(x-coef[0])@np.linalg.pinv(coef[1:5]);
 for _ in range(35):
  r=f(zz)@coef-x; j=np.stack([(f(zz+e)@coef-f(zz-e)@coef)/.00002 for e in np.eye(4)*.00001],axis=2)
  step=np.linalg.solve(j+1e-8*np.eye(4),r[...,None])[...,0]; zz-=np.clip(step,-.2,.2)
 return zz,np.max(abs(f(zz)@coef-x))
AA,BB=pickle.load(open('/tmp/mapfits','rb'))
As=[]
for kind,base,h in [(k,b,h) for _,k,b,h,_ in AA[:7]]:
 def f(x):return aff(np.c_[x,feats(x,kind,h)] if base else feats(x,kind,h))
 cf=ls(f(U[good]),V[good]); zz,err=inverse(f,cf,X);print('inverse',kind,err,np.min(zz),np.max(zz),flush=True);As.append((kind,zz))
 if kind in ['sin','quint']:print('coeff',kind,np.round(cf,4))
# Sparse sinusoidal / rational calibration: inspect shared coefficients and try richer diagonal corrections.
for fk,init in [(lambda x,h:np.c_[np.sin(h[0]*x),np.sin(h[1]*x)],[2.3,4.1]),(lambda x,h:np.c_[np.sin(h[0]*x),x/(1+h[1]**2*x*x)],[2.4,.7]),(lambda x,h:np.c_[np.sin(h[0]*x),np.cbrt(x)],[2.4]),(lambda x,h:np.c_[np.sin(h[0]*x),np.tanh(h[1]*x)],[2.4,1.1])]:
 def f(x,h):return aff(np.c_[x,fk(x,h)])
 def res(h):
  a=f(U[good],h);return (a@ls(a,V[good])-V[good]).ravel()
 q=least_squares(res,init,max_nfev=150); ff=lambda x:f(x,q.x); cf=ls(ff(U[good]),V[good]); zz,err=inverse(ff,cf,X);print('new A',q.x,np.std(q.fun),err,np.round(cf,4));As.append((str(q.x),zz))
bs=[('raw',lambda x:x)]
for _,k,b,h,c in BB[:10]:
 bs.append((k,lambda x,k=k,b=b,h=h,c=c:aff(np.c_[x,feats(x,k,h)] if b else feats(x,k,h))@c))
bs.append(('affineB',lambda x:aff(x)@ls(aff(u),v)))
results=[]
for an,zz in As:
 for bn,fun in bs:
  b=fun(zz); d0=A['_even'](zz)@dfit;d=np.tanh(1.25*d0);c0=pair(b)@cfit;c=np.arcsinh(1.25*c0)
  for wn,w in [('one',b),('b+tanh',np.tanh(2.15*b)),('asinhB',np.arcsinh(2.15*b)),('sinhB',np.sinh(1.25*b)),('c0',c0),('c',c),('c215',np.arcsinh(2.15*c0)),('c0tanh',np.tanh(2.15*c0)),('d',d),('d0',d0),('v1+v2',np.c_[c,d]),('b+d',np.c_[b,d]),('sinB',np.sin(2.15*b))]:
   sc=cv(w);results.append((max(sc[0]),an,bn,wn,w))
for sc,an,bn,wn,w in sorted(results,key=lambda t:t[0])[:20]:print('SCORE',an,bn,wn,cv(w,5))
