from probe21 import *
from scipy.optimize import minimize_scalar
A=s['panel_a'];B=s['panel_b'];C=s['panel_c'];
def rms(e):return np.sqrt(np.mean(e*e))
def robust(f,v):
 k=np.ones(len(f),bool)
 for _ in range(16):
  w=reg(f[k],v[k]); e=np.linalg.norm(f@w-v,axis=1); nk=e<max(.001,3*np.median(e));
  if nk.sum()<f.shape[1]+2 or np.array_equal(k,nk):break
  k=nk
 return w,k
forms={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'erf':erf,'sinh':np.sinh}
for name,h in forms.items():
 def er(a):
  u,v=A['u'],A['v']; f=D(u,h(a*u),cyc(u));w,k=robust(f,v);return rms((f@w-v)[k])
 o=minimize_scalar(er,bounds=(.15,3.2),method='bounded');print('A',name,o.x,o.fun)
 if o.fun<.004:
  f=D(A['u'],h(o.x*A['u']),cyc(A['u']));print(np.round(robust(f,A['v'])[0],4))
def bfeat(u,a,mode):
 h=np.sin(a*u); c=np.cos(a*u)
 if mode=='quad':return D(h,h*h,pair(h))
 if mode=='cosquad':return D(h,c,pair(h))
 if mode=='cyc':return D(h,c,cyc(h))
 if mode=='cyclin':return D(h,c,cyc(u))
 if mode=='quadlin':return D(h,c,pair(u))
 if mode=='phase':return D(h,c,cyc(h),cyc(c))
 if mode=='sumdiff':return D(h,c,np.cos(a*(u+np.roll(u,-1,axis=1))),np.sin(a*(u+np.roll(u,-1,axis=1))))
for mode in ('quad','cosquad','cyc','cyclin','quadlin','phase','sumdiff'):
 def er(a):
  f=bfeat(B['u'],a,mode); return rms(f@reg(f,B['v'])-B['v'])
 o=minimize_scalar(er,bounds=(.4,2.5),method='bounded');print('B',mode,o.x,o.fun,er(1.25),er(2.15));
 if o.fun<.025:print(np.round(reg(bfeat(B['u'],o.x,mode),B['v']),4))
for label,g,v in [('C1',lambda u:D(u,cyc(u),u**3),C['v1']),('C2',lambda u:D(np.sin(.22*u),np.cos(.22*u)),C['v2'])]:
 f=g(C['u']); w=reg(f,v);print(label,rms(f@w-v),np.round(w,4))
