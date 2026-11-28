import pickle, importlib.util
import numpy as np
from scipy.optimize import minimize, minimize_scalar
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,z=(tr[k] for k in ('x','y','traj')); t=s['eval_times']; n=len(x)
aff=m._affine; quad=m._quadratic
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
a=s['panel_a']; off,gain=m._calibration(a); q=(x-off)/gain; print('keys', {k:sorted(v) if isinstance(v,dict) else v.shape for k,v in s.items()}); print('calibration',off,gain,'per-channel outliers',np.sum(abs(a['v']-a['u']*gain-off)>.002,axis=0)); print('times',t)
pb=s['panel_b']; u,v=pb['u'],pb['v']; out={k:[] for k in ('raw_quadratic','full_sine_quadratic','diagonal_sine')}; freq=[]
for i in range(len(u)):
 k=np.arange(len(u))!=i
 for name in out:
  if name=='raw_quadratic': F=quad(u)
  else:
   f=(lambda h:quad(np.sin(h*u))) if name=='full_sine_quadratic' else (lambda h:aff(np.sin(h*u)**2))
   def loss(h):
    a=f(h)[k]; return np.sum((a@ls(a,v[k])-v[k])**2)
   h=minimize_scalar(loss,bounds=(1.05,1.55),method='bounded').x; F=f(h)
   if name=='diagonal_sine': freq.append(h)
  out[name].append(F[i]@ls(F[k],v[k])-v[i])
print('panel B true leave-one-out RMSE', {k:np.sqrt(np.mean(np.array(v)**2)) for k,v in out.items()}, 'frequency range',min(freq),max(freq))
b=m._operator(pb,q); F=quad(b); H=aff(b); z2=H@ls(H,z)
print('quadratic y coefficient matrix order upper triangle',ls(F[:52],y[:52]).round(5)); print('trajectory residual RMSE by timestamp',np.sqrt(np.mean((z-z2)**2,0)))
i,j=np.triu_indices(4,1); Q=np.c_[b,b[:,i]+b[:,j],b[:,i]-b[:,j],b[:,i]+.5*b[:,j],b[:,i]-.5*b[:,j]]
def challenger(f,x,y,train,test):
 # Select one rational quotient using only the outer training set, in inner folds.
 inner=np.array_split(train,4); objects=[(0,None,f)]
 for a in (.03,.1,.3,1.):
  for col in range(x.shape[1]): objects.append((a,col,f/(1+a*x[:,col,None]**2)))
 def score(g):
  return np.mean([np.mean(abs(g[iv]@ls(g[np.setdiff1d(train,iv)],y[np.setdiff1d(train,iv)])-y[iv])) for iv in inner])
 a,j,g=min(objects,key=lambda row:score(row[2])); return g[test]@ls(g[train],y[train]),(a,j)
def timefit(k,ii,kind):
 def temp(p):
  a,c=np.exp(p); return np.column_stack((-np.expm1(-a*t), -np.expm1(-c*t) if kind=='exp' else t/(1+c*t)))
 def solve(T): return ls(H[k],z[k]@np.linalg.pinv(T.T))
 def loss(p):
  T=temp(p); return np.sum((H[k]@solve(T)@T.T-z[k])**2)
 rate=minimize(loss,np.log([1.8,.4]),method='L-BFGS-B',bounds=[(-4,4),(-4,4)]).x; T=temp(rate); return H[ii]@solve(T)@T.T
splits=[np.arange(52,72),np.arange(20),np.arange(26,46),np.argsort(np.linalg.norm(x,axis=1))[-20:]]+[np.argsort(sign*x[:,kk])[-20:] for kk in range(4) for sign in (-1,1)]
r=[]
for ii in splits:
 k=np.setdiff1d(np.arange(n),ii); base_y=F[ii]@ls(F[k],y[k]); yr,pick=challenger(F,Q,y,k,ii); modes=np.linalg.svd(z[k],full_matrices=False)[2][:2]; base_z=(H[ii]@ls(H[k],z[k]@modes.T))@modes
 def ez(pred): return abs(pred-z[ii]).mean()
 r.append([abs(base_y-y[ii]).mean(),abs(yr-y[ii]).mean(),ez(base_z),ez(timefit(k,ii,'exp')),ez(timefit(k,ii,'rat'))]); print('outer',len(r),np.round(r[-1],6),'rational choice',pick)
print('worst',np.max(r,axis=0)); print('pairwise algebra rank',np.linalg.matrix_rank(F),np.linalg.matrix_rank(np.c_[F,Q,Q**2]))
# There are no batch tags or categorical outcomes. Probe continuous residuals by row blocks as a check for drift.
py=F@ls(F,y); hy=np.linalg.qr(F)[0]; ey=(y-py)/(1-(hy**2).sum(1)); hz=np.linalg.qr(H)[0]; ez=(z-z2)/(1-(hz**2).sum(1))[:,None]; print('row-block LOO mean residuals',[(float(ey[k].mean()),float(ez[k].mean()),float(ez[k,-1].mean())) for k in np.array_split(np.arange(n),4)])
