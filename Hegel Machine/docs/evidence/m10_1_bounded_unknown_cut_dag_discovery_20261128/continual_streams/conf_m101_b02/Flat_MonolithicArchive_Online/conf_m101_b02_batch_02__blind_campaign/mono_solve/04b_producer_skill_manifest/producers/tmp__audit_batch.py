import importlib.util, pickle
import numpy as np
from itertools import combinations, combinations_with_replacement
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; z=np.c_[y,tr]; f=m._design(x)
print('Observed fields:', sorted(s), 'train:',sorted(s['train']))
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def affine(u): return np.c_[np.ones(len(u)),u]
p=s['panel_a']; h=affine(p['u']); keep=np.ones(len(h),bool)
for i in range(8):
 c=ls(h[keep],p['v'][keep]); e=np.linalg.norm(h@c-p['v'],axis=1); keep=e<max(.004,np.sort(e)[34])
c=ls(h[keep],p['v'][keep]); p=s['panel_b']; b=ls(affine(p['u']),p['v'])
p=s['panel_c']; q=ls(m._design(p['u'])[:,:15],p['v1'])[5:]; terms=list(combinations_with_replacement(range(4),2)); support=[pair for pair,row in zip(terms,q) if np.linalg.norm(row)>.01]; print('Panel C v1 quadratic support:',support)
def forward_a(u): return affine(u)@c
def inverse_a(u): return (u-c[0])@np.linalg.inv(c[1:])
def forward_b(u): return affine(u)@b
models={}
for name,u in [('raw',x),('A',forward_a(x)),('Ainv',inverse_a(x)),('B',forward_b(x)),('B(A)',forward_b(forward_a(x))),('B(Ainv)',forward_b(inverse_a(x)))]:
 quad=np.stack([u[:,i]*u[:,j] for i,j in support],axis=1); cubic=(quad[:,:,None]*u[:,None,:]).reshape(len(x),-1)
 candidates=np.c_[f[:,:15],cubic]; _,sing,v=np.linalg.svd(candidates,full_matrices=False); models[name]=(candidates@v[sing>1e-9].T)
# Probe fixed rational denominators including all unit pair sums and differences.
rat={}
for i,j in [(i,None) for i in range(4)]+list(combinations(range(4),2)):
 for sign in ([1] if j is None else [-1,1]):
  w=x[:,i] if j is None else (x[:,i]+sign*x[:,j])/np.sqrt(2)
  for a in (.025,.1,.4): rat[str((i,j,sign,a))]=f/(1+a*w[:,None]**2)
f4=np.c_[f,np.column_stack([x[:,p].prod(1) for p in combinations_with_replacement(range(4),4)])]
def coordinates(a,mode):
 if mode=='exp':
  def basis(r): return -np.expm1(-np.outer(t,r))
  def resid(r):
   d=basis(r); return (d@ls(d,tr[a].T)-tr[a].T).ravel()
  d=basis(least_squares(resid,[1.8,.4],bounds=([.8,.05],[5,.79])).x); v=np.linalg.qr(d)[0].T
 else: v=np.linalg.svd(tr[a],full_matrices=False)[2][:2]
 return np.c_[y[a],tr[a]@v.T],v
def pred(g,a,b,mode='svd'):
 zz,v=coordinates(a,mode); h=g[b]@ls(g[a],zz); return np.c_[h[:,0],h[:,1:]@v]
def metric(p,b):
 e=np.abs(p-z[b]); return np.r_[e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()]
def select(pool,a):
 inner=[(np.setdiff1d(a,v),v) for v in np.array_split(np.random.default_rng(101).permutation(a),3)]
 def score(g):
  return np.mean([metric(pred(g,j,k),k)[:2].mean() for j,k in inner])
 return min(pool,key=lambda key:score(pool[key]))
r=np.argsort(np.linalg.norm(x,axis=1)); folds=[('tail52',np.arange(52),np.arange(52,72)),('radial',r[:52],r[52:])];
for i,v in enumerate(np.array_split(np.random.default_rng(2027).permutation(72),4)): folds.append((f'fold{i}',np.setdiff1d(np.arange(72),v),v))
for name,a,b in folds:
 print(name,'scalar / trajectory / worst horizon MAE'); out=m.solve_batch(dict(s, train=dict(x=x[a],y=y[a],traj=tr[a]),test=dict(x=x[b]))); print('artifact',np.round(metric(np.c_[out['y_hat'],out['state_pred']],b),5)); print('quadratic',np.round(metric(pred(f[:,:15],a,b),b),5)); print('cubic + exponential modes',np.round(metric(pred(f,a,b,'exp'),b),5))
 for label,pool in [('panel support',models),('rational',rat)]:
  k=select(pool,a); print(label,k,np.round(metric(pred(pool[k],a,b),b),5))
 if name in ('tail52','radial'):
  print('panel candidates',[(k,np.round(metric(pred(g,a,b),b),5).tolist()) for k,g in models.items()]); h=np.linalg.lstsq(f[a], f[b].T.T if False else z[a],rcond=None)[0]; err=z[b]-f[b]@h; print('cubic fit held-out mean residual by channel',np.round(err.mean(0),5))
