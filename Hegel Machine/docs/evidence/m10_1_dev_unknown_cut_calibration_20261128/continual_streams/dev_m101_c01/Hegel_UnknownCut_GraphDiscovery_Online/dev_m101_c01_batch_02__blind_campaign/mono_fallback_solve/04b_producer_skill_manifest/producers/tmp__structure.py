import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement, product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def affine(p):
 u,v=p['u'],p['v']; f=np.c_[np.ones(len(u)),u]; keep=np.ones(len(u),bool)
 for _ in range(8):
  w=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0]; r=np.linalg.norm(v-f@w,axis=1); keep=r<max(.001,4*np.median(r))
 return w
def apply(x,w):return x@w[1:]+w[0]
a=affine(s['panel_a']);b=affine(s['panel_b']);xa=apply(x,a); xb=apply(x,b); ab=apply(xa,b); ba=apply(xb,a)
step=lambda k:-np.expm1(-k*t)
wave=lambda d,w:np.exp(-d*t)*np.sin(w*t)
phase=lambda d,w:1-np.exp(-d*t)*np.cos(w*t)
types=[('step-wave',lambda q:np.c_[step(q[0]),wave(*q[1:])],[.55,.18,1.15]),('tanh-wave',lambda q:np.c_[np.tanh(q[0]*t),wave(*q[1:])],[.5,.2,1.1]),('t-wave',lambda q:np.c_[t,wave(*q)],[.3,1]),('step-phase',lambda q:np.c_[step(q[0]),phase(*q[1:])],[.5,.4,.8]),('paired-osc',lambda q:np.c_[phase(*q),wave(*q)],[.3,1.1]),('wave-wave',lambda q:np.c_[wave(*q[:2]),wave(*q[2:])],[.2,.2,.2,1.1]),('wave-t_exp',lambda q:np.c_[t*np.exp(-q[0]*t),wave(*q[1:])],[.5,.2,1.1]),('wave-mixed_step',lambda q:np.c_[step(q[0])*np.exp(-q[1]*t),wave(*q[2:])],[.5,.01,.2,1.1])]
def temporal(z):
 out=[]
 for name,fun,p in types:
  def resid(q):
   f=fun(q); return (f@np.linalg.lstsq(f,z.T,rcond=None)[0]-z.T).ravel()
  opt=least_squares(resid,p,bounds=(0,5),max_nfev=500); out.append((np.sqrt(np.mean(opt.fun**2)),name,opt.x,fun(opt.x)))
 return sorted(out,key=lambda q:q[0])
def poly(u,k=2):return np.column_stack([np.ones(len(u))]+[np.prod(u[:,p],axis=1) for d in range(1,k+1) for p in combinations_with_replacement(range(4),d)])
if __name__=='__main__':
 best=temporal(z[:56]);
 for e,name,q,f in best:
  print(name,q,'fit',round(e,6),'held-out projection',round(np.mean(abs(f@np.linalg.lstsq(f,z[56:].T,rcond=None)[0]-z[56:].T)),6))
 f=next(f for e,name,q,f in best if name=='step-wave'); c=np.linalg.lstsq(f,z.T,rcond=None)[0].T; Y=np.c_[y,c]; print('coefficient stats',c.mean(0),c.std(0))
 for nm,u in [('raw',x),('A',xa),('B',xb),('AB',ab),('BA',ba)]:
  p=poly(u); w=np.linalg.lstsq(p,Y,rcond=None)[0]; print(nm,'quadratic rows 1; 0-3; 00,01,02,03,11,12,13,22,23,33',w.T)
 for nm,u in [('raw',x),('A',xa),('B',xb),('AB',ab),('BA',ba)]:
  F=[]; names=[]
  for j in range(4):
   a0=u[:,j]
   for w in [.5,1.,1.5,2.,2.5,3.]:
    for fn in (np.sin,np.cos,np.tanh):F.append(fn(w*a0));names.append((fn.__name__,w,j))
  for j,k in product(range(4),repeat=2):
   a0,b0=u[:,j],u[:,k]
   for name,v0 in [('sinlin',np.sin(a0)*b0),('tanhlin',np.tanh(a0)*b0),('frac',a0/(1+b0*b0)),('fracprod',a0*b0/(1+a0*a0)),('abs',abs(a0)*b0),('sumwave',np.sin(a0+b0)),('diffwave',np.sin(a0-b0)),('cub',a0**2*b0)]: F.append(v0);names.append((name,j,k))
  F=np.array(F).T
  p=poly(x); w=np.linalg.lstsq(p[:56],np.c_[Y,F][:56],rcond=None)[0]; r=np.c_[Y,F]-p@w; u0=r[:56,3:]; d=(u0*u0).sum(0); beta=(u0.T@r[:56,:3])/(d[:,None]+1e-28); gain=abs(beta)*np.sqrt(d[:,None]); print(nm,'top residual features:')
  for j in range(3):
   top=np.argsort(gain[:,j])[-4:][::-1]; print(j,[(names[i],round(gain[i,j]/np.linalg.norm(r[:56,j]),3),round(np.mean(abs(r[56:,j]-r[56:,i+3]*beta[i,j])),4)) for i in top])
