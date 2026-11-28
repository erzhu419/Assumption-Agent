import numpy as np,pickle,sys
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
from conf_m102_a04_batch_00_solver import _align,lin,one
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_a'];u,v=p['u'],p['v']; a,b=_align(p); r=v-(a+b*u); print('diagonal affine calibration',a,b,np.sort(np.linalg.norm(r,axis=1))); keep=np.linalg.norm(v,axis=1)<4; print('full affine residual',np.std(v[keep]-one(u[keep])@lin(one(u[keep]),v[keep])))
p=s['panel_c'];u=p['u']; z=(np.r_[s['train']['x'],s['test']['x']]-a)/b

def prof(fn,Y,q,bounds):
 def r(q):
  f=fn(u,q);return (f@lin(f,Y)-Y).ravel()
 q=least_squares(r,q,bounds=bounds,gtol=1e-13).x;w=lin(fn(u,q),Y);return q,np.std(r(q)),fn(z,q)@w

g=lambda x,q:one(x/(1+q[0]*abs(x))+q[1]*x)
h=lambda x,q:np.c_[one(x),x*x,x*x/(1+q[0]*x*x)]
for name,fn,key,q,bnd in [('signed',g,'v1',[.85,-.01],([.001,-1],[20,1])),('curved',h,'v2',[.25],(.001,5)),('tanh',lambda x,q:one(np.tanh(q[0]*x)+q[1]*x),'v1',[1.,.0],([.1,-1],[8,1])),('rational odd',lambda x,q:one(x/(1+q[0]*x*x)+q[1]*x),'v1',[1.,.0],([.001,-1],[8,1]))]:
 q,r,_=prof(fn,p[key],q,bnd);print(name,q,r)
t=s['eval_times']; Y=s['train']['traj'][:52].T
f=lambda r:-np.expm1(-t[:,None]*r)
def resid(r):
 B=f(r);return (B@lin(B,Y)-Y).ravel()
q=least_squares(resid,[.4,1.8],bounds=([.05,.8],[.8,4]),gtol=1e-13).x;print('temporal rates',q,'residual',np.std(resid(q)),'mode std',lin(f(q),Y).std(1))
