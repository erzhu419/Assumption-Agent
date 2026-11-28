import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; n=len(x)
np.set_printoptions(precision=5,suppress=True,linewidth=130)
def affine(p):
 u,v=p['u'],p['v']; f=np.c_[np.ones(len(u)),u]; keep=np.ones(len(u),bool)
 for _ in range(8):
  w=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0]; r=np.linalg.norm(v-f@w,axis=1); keep=r<max(0.001,4*np.median(r))
 return w,r
A,ra=affine(s['panel_a']); B,rb=affine(s['panel_b']); print('t/tel/summary',t,s['telemetry_vec'],s['train']['summary']);print('A',A,'residuals',np.sort(ra));print('B',B)
def ap(x,w):return x@w[1:]+w[0]
xa=ap(x,A);xb=ap(x,B);ab=ap(xa,B);ba=ap(xb,A)
def varproj(fun,p):
 def fit(q):
  f=fun(q); a=np.linalg.lstsq(f,z.T,rcond=None)[0]; return (f@a-z.T).ravel()
 opt=least_squares(fit,p,max_nfev=1200); return opt.x,np.sqrt(np.mean(opt.fun**2))
for name,fun,p in [('double_step',lambda q:-np.expm1(-t[:,None]*np.exp(q)),[.4,-1.6]),('step_damped',lambda q:np.c_[-np.expm1(-np.exp(q[0])*t), t*np.exp(-np.exp(q[1])*t)],[0,-.4]),('step_sine',lambda q:np.c_[-np.expm1(-np.exp(q[0])*t),np.sin(np.exp(q[1])*t)],[0,0]),('step_tanh',lambda q:np.c_[-np.expm1(-np.exp(q[0])*t),np.tanh(np.exp(q[1])*t)],[0,0]),('double_tanh',lambda q:np.tanh(t[:,None]*np.exp(q)),[0,-1]),('exp_sine',lambda q:np.c_[-np.expm1(-np.exp(q[0])*t),np.exp(-np.exp(q[1])*t)*np.sin(np.exp(q[2])*t)],[0,-1,0]),('step_cos',lambda q:np.c_[-np.expm1(-np.exp(q[0])*t),1-np.cos(np.exp(q[1])*t)],[0,0])]:
 q,e=varproj(fun,p); print(name,np.exp(q),e)
U,D,V=np.linalg.svd(z,full_matrices=False); targets=np.c_[y,U[:,:2]*np.sqrt(n)];print('rank2 noise RMSE',np.sqrt(np.mean((z-U[:,:2]@np.diag(D[:2])@V[:2])**2)))
def poly(u,d):return np.column_stack([np.ones(len(u))]+[np.prod(u[:,p],axis=1) for k in range(1,d+1) for p in combinations_with_replacement(range(4),k)])
for name,u in [('X',x),('A',xa),('B',xb),('AB',ab),('BA',ba)]:
 print(name)
 for d in range(1,5):
  f=poly(u,d); Q=np.linalg.qr(f,mode='reduced')[0]; r=targets-Q@(Q.T@targets); print(d,'56/16',np.abs(f[56:]@np.linalg.lstsq(f[:56],targets[:56],rcond=None)[0]-targets[56:]).mean(0),'loo',np.mean(abs(r/(1-(Q*Q).sum(1))[:,None]),axis=0))
 f=np.c_[u,np.tanh(u),np.sin(u),u**2,np.sin(2*u),np.cos(u)]; f=f-f.mean(0); print('top marginal',[(j,[(int(i),round(c[i],3)) for i in np.argsort(abs(c))[-4:]]) for j in range(3) for c in [f.T@(targets[:,j]-targets[:,j].mean())/(np.linalg.norm(f,axis=0)*np.std(targets[:,j])*np.sqrt(n))]])
