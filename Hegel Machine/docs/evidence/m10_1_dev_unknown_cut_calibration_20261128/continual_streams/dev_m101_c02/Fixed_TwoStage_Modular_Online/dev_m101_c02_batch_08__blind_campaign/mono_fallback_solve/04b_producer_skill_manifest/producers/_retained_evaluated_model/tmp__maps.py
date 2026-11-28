from probe import s,ls,poly
import numpy as np
from scipy.optimize import lsq_linear,least_squares,minimize_scalar

def bias(x):return np.c_[np.ones(len(x)),x]
def calfit(s):
 p=s['panel_a'];u=p['u'];v=p['v'];g=np.max(np.abs(v-u),axis=1)<2;u=u[g]; v=v[g];z=np.roll(u,-1,axis=1)
 return np.array([ls(np.c_[np.ones(len(u)),u[:,j],z[:,j],u[:,j]*z[:,j],u[:,j]**3],v[:,j]) for j in range(4)]).T

def cal(x,c):
 z=np.roll(x,-1,axis=1);return c[0]+c[1]*x+c[2]*z+c[3]*x*z+c[4]*x**3

def inv(x,c):
 z=x.copy();j=np.arange(4)
 for _ in range(12):
  J=np.zeros((len(z),4,4));J[:,j,j]=c[1]+c[3]*np.roll(z,-1,axis=1)+3*c[4]*z*z;J[:,j,(j+1)%4]=c[2]+c[3]*z
  z-=np.linalg.solve(J,(cal(z,c)-x)[...,None])[...,0]
 return z

funcs={'sin':np.sin,'tanh':np.tanh,'sinh':np.sinh,'cos':np.cos,'cosh':np.cosh,'atan':np.arctan,'cubic':lambda x:x**3,'sin2':lambda x:np.sin(x)**2,'tanh2':lambda x:np.tanh(x)**2,'sq':lambda x:x*x,'sin_cubed':lambda x:np.sin(x)**3,'tanh_cubed':lambda x:np.tanh(x)**3}

def fitpanel(u,v):
 ranks=[]
 for name,fn in funcs.items():
  for lin in [False,True]:
   def design(x,k):return bias(np.c_[x,fn(k*x)] if lin else fn(k*x))
   def err(k):
    f=design(u,k); return np.mean((f@ls(f,v)-v)**2)
   grid=np.linspace(.12,6,55); scores=np.array([err(k) for k in grid]);i=scores.argmin();r=minimize_scalar(err,bounds=(grid[max(i-1,0)],grid[min(i+1,len(grid)-1)]),method='bounded');k=r.x
   f=design(u,k); coef=ls(f,v); H=np.sum(np.linalg.qr(f,mode='reduced')[0]**2,axis=1); cv=np.mean(np.abs((f@coef-v)/(1-H[:,None])));ranks.append((cv,name,lin,k,r.fun,lambda x,d=design,k=k,c=coef:d(x,k)@c))
 return sorted(ranks,key=lambda q:q[0])

if __name__=='__main__':
 maps={};c=calfit(s);print('cal',c)
 for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  p=s[pn];rs=fitpanel(p['u'],p[vn]); print(pn,vn,[r[:5] for r in rs[:6]]);maps[vn]=rs[0][-1]
 tr=s['train'];x=tr['x'];t=s['eval_times'];T=tr['traj'];Y=tr['y']; non=np.linalg.svd(T,full_matrices=False)[2][0];
 for name,fun in [('exp',lambda t,k:-np.expm1(-k*t)),('tanh',lambda t,k:np.tanh(k*t)),('rat',lambda t,k:t/(1+k*t))]:
  def obj(k):
   b=fun(t,k)[:,None];return np.std(T.T-b@ls(b,T.T))
  r=minimize_scalar(obj,bounds=(.001,5),method='bounded');print('time',name,r.x,r.fun)
 names={'raw':x,'inv':inv(x,c),'fwd':cal(x,c)}; out=[]
 for name,X in names.items():
  print(name,'range',X.min(),X.max())
  for op in ['', 'v','v1','v2','v:v1','v:v2','v1:v2','v1:v','v2:v']:
   z=X
   for o in op.split(':') if op else []:z=maps[o](z)
   for n,F in [('lin',bias(z)),('sq',poly(z,2)[0]),('sin',bias(np.sin(z))),('tanh',bias(np.tanh(z))),('lin+tanh',bias(np.c_[z,np.tanh(z)]))]:
    coef=ls(F[:56],np.c_[Y,T][:56]);e=np.abs(F[56:]@coef-np.c_[Y,T][56:]).mean(axis=0);out.append((e[0],e[1:].mean(),name,op,n))
 print('top y',sorted(out)[:15]);print('top traj',sorted(out,key=lambda v:v[1])[:15])
