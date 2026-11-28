import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda X,y:np.linalg.lstsq(X,y,rcond=None)[0]
def aff(z):return np.c_[np.ones(len(z)),z]
def rol(z,k=1):return np.roll(z,-k,axis=1)
u,v=[s['panel_a'][k] for k in ['u','v']]; keep=np.linalg.norm(v,axis=1)<5; u,v=u[keep],v[keep]
funcs={'sin':(np.sin,.1,6),'cos':(np.cos,.1,6),'tanh':(np.tanh,.1,6),'rat':(lambda z:z/(1+z*z),.05,5),'atan':(np.arctan,.05,5),'exp':(np.exp,.05,2.5)}
trans={'u':lambda z:z,'next':rol,'next2':lambda z:rol(z,2),'prev':lambda z:rol(z,3),'prod':lambda z:z*rol(z),'sum':lambda z:z+rol(z),'diff':lambda z:z-rol(z),'sum2':lambda z:z+rol(z,2),'diff2':lambda z:z-rol(z,2)}
# Compare one shared nonlinear term across cyclic outputs, with diagonal/next bilinear retained.
best=[]
for tn,tf in trans.items():
 for fn,(f,lo,hi) in funcs.items():
  def calc(w):
   zz=f(w*tf(u)); co=[]; errs=[]
   for j in range(4):
    h=np.c_[aff(u),u[:,j]*u[:,(j+1)%4],zz[:,j]]; c=ls(h,v[:,j]); co.append(c); errs.append(h@c-v[:,j]);
   return np.linalg.norm(errs)/np.sqrt(v.size),np.array(co)
  grid=np.linspace(lo,hi,30); i=np.argmin([calc(w)[0] for w in grid]); q=minimize_scalar(lambda w:calc(w)[0],bounds=(grid[max(0,i-1)],grid[min(29,i+1)]),method='bounded');best.append((q.fun,tn,fn,q.x,calc(q.x)[1]))
for e,tn,fn,w,c in sorted(best,key=lambda a:a[0])[:8]:print('A',e,tn,fn,w,np.round(c,5))
for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u1=s[k]['u']; v1=s[k][j]; out=[]
 for fn,(f,lo,hi) in funcs.items():
  for linear in [False,True]:
   def calc(w):
    A=aff(np.c_[u1,f(w*u1)] if linear else f(w*u1)); c=ls(A,v1); return np.linalg.norm(A@c-v1)/np.sqrt(v1.size),c
   grid=np.linspace(lo,hi,35); i=np.argmin([calc(w)[0] for w in grid]); q=minimize_scalar(lambda w:calc(w)[0],bounds=(grid[max(0,i-1)],grid[min(34,i+1)]),method='bounded');out.append((q.fun,fn,linear,q.x,calc(q.x)[1]))
 for e,fn,lin,w,c in sorted(out,key=lambda a:a[0])[:3]:print(k,j,e,fn,lin,w,np.round(c,5))
