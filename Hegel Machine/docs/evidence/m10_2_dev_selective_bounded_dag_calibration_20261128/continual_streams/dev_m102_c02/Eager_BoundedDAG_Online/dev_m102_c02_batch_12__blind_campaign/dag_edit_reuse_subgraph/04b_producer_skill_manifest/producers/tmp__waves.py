import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from probe import *
from scipy.optimize import least_squares
from itertools import product

def rms(q):return np.sqrt(np.mean(q*q))
def basis(x,w,k):
 p=x*w
 if k=='cos': f=np.cos(p)
 if k=='sin': f=np.sin(p)
 if k=='both': f=np.c_[np.cos(p),np.sin(p)]
 if k=='coslin': f=np.c_[np.cos(p),x]
 if k=='sinlin': f=np.c_[np.sin(p),x]
 if k=='rational': f=1/(1+p*p)
 if k=='tanh': f=np.tanh(p)
 if k=='sq': f=p*p
 if k=='abs': f=np.abs(x)
 return aff(f)
def fit(x,v,k,start):
 def fun(w):return res(basis(x,w,k),v).ravel()
 z=least_squares(fun,start,bounds=(.04,14),max_nfev=120)
 return rms(z.fun),z.x,ls(basis(x,z.x,k),v)
X=(T['x']-o)/g
for name,x,v in [('B',b['u'],b['v']),('C1',c['u'],c['v1']),('C2',c['u'],c['v2']),('train',X,np.c_[T['y'],T['traj']])]:
 print(name)
 for k in ('sq','abs','cos','sin','both','coslin','sinlin','rational','tanh'):
  err,w,B=min([fit(x,v,k,np.full(4,i)) for i in (0.25,1,2.15,3.14,5,7,10)],key=lambda q:q[0]);print(k,err,w)
  if name=='B' and k in ('cos','both','coslin'):
   f=basis(X,w,k)@B; print('  coeff',B);print('  train rmse',rms(res(aff(f),T['traj'])),rms(res(np.c_[aff(f),np.sum(f[:,:2]*f[:,2:],1)],T['y'])))
