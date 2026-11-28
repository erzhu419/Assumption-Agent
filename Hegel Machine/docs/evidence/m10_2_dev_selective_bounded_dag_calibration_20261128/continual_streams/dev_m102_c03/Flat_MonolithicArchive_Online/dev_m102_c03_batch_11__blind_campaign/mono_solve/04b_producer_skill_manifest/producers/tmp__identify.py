import numpy as np,pickle
from scipy.optimize import least_squares, minimize_scalar
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(u):return np.c_[np.ones(len(u)),u]
func={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'gauss':lambda x:np.exp(-x*x),'rat':lambda x:x/(1+x*x),'evenrat':lambda x:1/(1+x*x),'atan':np.arctan}
def loss(f,y):return np.sqrt(np.mean((f@ls(f,y)-y)**2))
p=s['panel_a'];u=p['u'];v=p['v'];a=aff(u);good=np.linalg.norm(v-a@ls(a,v),axis=1)<3;u=u[good];v=v[good];a=aff(u)
for j in range(4):
 x=u[:,j]; w=u[:,(j+2)%4];y=v[:,j];out=[]
 for i in range(4):
  z=u[:,i];
  for name,fn in func.items():
   for base in [a,np.c_[a,x*w,x**3],np.c_[a,x*w,w**2,x**2,x**3,w**3]]:
    f=lambda q:loss(np.c_[base,fn(q*z)],y);r=minimize_scalar(f,bounds=(.12,4),method='bounded');out.append((r.fun,i,name,round(r.x,4),base.shape[1]))
 print('A',j,'poly',loss(np.c_[a,x*w,x*x,w*w,x**3,w**3],y),'best',sorted(out)[:6]);
p=s['panel_c'];u=p['u'];a=aff(u)
for key in ['v1','v2']:
 y=p[key];out=[]
 for name,fn in func.items():
  for extra in [lambda u:np.ones(len(u)),lambda u:u,lambda u:u*u,lambda u:u**3,lambda u:np.c_[u,u*u],lambda u:np.c_[u,u**3],lambda u:np.c_[u,u*u,u**3]]:
   b=np.c_[np.ones(len(u)),extra(u)];f=lambda q:loss(np.c_[b,fn(q*u)],y);r=minimize_scalar(f,bounds=(.15,4),method='bounded');out.append((r.fun,name,round(r.x,5),b.shape[1],f(2.15)))
 print('C',key,sorted(out)[:12])
t=s['eval_times'];v=s['train']['traj'][:52].T
for tag,f in [('exp pair',lambda r:-np.expm1(-t[:,None]*r)),('exp ramp',lambda r:np.c_[t,-np.expm1(-t*r[0])]),('decay rise',lambda r:np.c_[t*np.exp(-t*r[0]),-np.expm1(-t*r[1])])]:
 def res(r):
  a=f(r);return (a@ls(a,v)-v).ravel()
 o=least_squares(res,[.35,1.8],bounds=(.02,5));print('time',tag,o.x,np.std(res(o.x)))
