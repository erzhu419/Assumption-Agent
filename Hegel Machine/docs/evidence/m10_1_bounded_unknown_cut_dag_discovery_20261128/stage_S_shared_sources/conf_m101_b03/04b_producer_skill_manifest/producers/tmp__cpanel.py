from modes import *
from scipy.optimize import minimize_scalar
p=s['panel_c']; u=p['u']; v=p['v1']; w=p['v2']; ii=np.arange(len(u));
def fit_loo(F,z):
 b=ls(F,z); e=z-F@b; Q=np.linalg.svd(F,full_matrices=False)[0]; return np.mean(abs(e/np.maximum(1-np.sum(Q**2,axis=1)[:,None],1e-6))),np.std(e),b
for name,make in [('plain',lambda x:x),('B',lambda x:aff(x)@B),('Binv',lambda x:(x-B[0])@np.linalg.inv(B[1:]))]:
 U=make(u); G=PolynomialFeatures(2).fit_transform(U); print(name,'v2 full quadratic',fit_loo(G,w)[:2]);
 for extra in [False,True]:
  for fn in [np.tanh,np.sin,np.arctan,lambda z:z/(1+z*z)]:
   for mode in ['pre','post']:
    if mode=='post' and fn.__name__ not in ['tanh','sin']:continue
    if mode=='pre':
     def ff(k):return fit_loo(np.c_[aff(fn(U*k)),U if extra else np.empty((len(U),0))],v)
     lo,hi=.05,2.5
    else:
     def ff(k):
      inv=np.arctanh if fn.__name__=='tanh' else np.arcsin
      W=inv(v/k); F=G if extra else aff(U); L=ls(F,W); E=v-k*fn(F@L); Q=np.linalg.svd(F,full_matrices=False)[0]; return np.mean(abs(E/(1-np.sum(Q**2,axis=1)[:,None]))),np.std(E),L
     lo,hi=abs(v).max()*1.00001,3
    h=minimize_scalar(lambda k:ff(k)[0],bounds=(lo,hi),method='bounded'); print(name,extra,fn.__name__,mode,round(h.x,3),np.round(ff(h.x)[:2],5))
print('telemetry',s['telemetry_vec'],'summary',tr['summary']); print('B',B)
