import numpy as np
from scipy.optimize import least_squares,minimize_scalar

def aff(x):return np.c_[np.ones(len(x)),x]
def ls(d,y):return np.linalg.lstsq(d,y,rcond=None)[0]
def profile(u,v,d,init,lo,hi):
 def res(k):
  f=d(u,k); return (f@ls(f,v)-v).ravel()
 o=least_squares(res,init,bounds=(lo,hi),max_nfev=120); return o.x,ls(d(u,o.x),v),np.mean(o.fun**2)**.5

def coords(s,x,verbose=False):
 a,b,c=(s['panel_'+k] for k in 'abc'); u=a['u'];v=a['v'];m=np.linalg.norm(v-u,axis=1)<4; u,v=u[m],v[m]
 ca=lambda q,k:np.c_[aff(q),np.sin(k[0]*q)]
 ka,wa,ea=profile(u,v,ca,[2.4],[1.5],[3.2]);
 ka2,wa2,ea2=profile(u,v,lambda q,k:np.c_[aff(q),np.sin(k[0]*q+k[1:])],[2.4,0,0,0,0],[1.5,-1,-1,-1,-1],[3.2,1,1,1,1]);
 wa0=ls(aff(u),v); q=(x-wa0[0])@np.linalg.inv(wa0[1:]);
 for _ in range(30): q=(x-wa[0]-np.sin(ka[0]*q)@wa[5:])@np.linalg.inv(wa[1:5]);
 d=lambda q,k:aff(np.arctan(k[0]*q+k[1:]));kc,wc,ec=profile(c['u'],c['v1'],d,[2.15,0,0,0,0],[.5,-1,-1,-1,-1],[5,1,1,1,1]);
 d2=lambda q,k:aff(q+k[1]*np.log1p(k[0]*q*q));kd,wd,ed=profile(c['u'],c['v2'],d2,[.36,1.5],[.02,.2],[2,3]);
 wb=ls(aff(b['u']),b['v']);
 if verbose: print('panel A',ka,ea,ka2,ea2,wa.round(4),'C1',kc,ec,'C2',kd,ed, 'inv err',np.max(abs(ca(q,ka)@wa-x)),flush=True)
 op={'b':lambda p:aff(p)@wb,'c':lambda p:d(p,kc)@wc,'d':lambda p:d2(p,kd)@wd,'a':lambda p:ca(p,ka)@wa}
 out={'x':x,'q':q}
 for k in 'abcd':
  for j in ('x','q'):
   if k=='a' and j=='q':continue
   out[k+j]=op[k](out[j]);
 for k in 'bcd':
  for l in 'bcd':
   if k!=l:
    for j in ('x','q'):out[k+l+j]=op[k](out[l+j]);
 return out
