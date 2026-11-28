import pickle, numpy as np, importlib.util, itertools
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; n=len(y)
def load(p):
 sp=importlib.util.spec_from_file_location('mod',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m);return m
m=load('/holdout_task/_frozen_nodes/node_00.py'); arc=load('/tmp/hegel_continual_archive/conf_m102_b04_batch_00_solver.py'); F=m.run_node({'__input__':s}); np.set_printoptions(precision=4,suppress=True)
def affine(u):return np.c_[np.ones(len(u)),u]
def basis(u,d):return np.column_stack([np.ones(len(u))]+[np.prod(u[:,j],axis=1) for i in range(1,d+1) for j in itertools.combinations_with_replacement(range(u.shape[1]),i)])
Q={}; matrices={}
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u'];f=affine(u)
 for g,v in p.items():
  if g=='u':continue
  w=np.linalg.lstsq(f,v,rcond=None)[0]; e=np.linalg.norm(f@w-v,axis=1)
  print(k,g,'affine',w,'residual quantiles',np.quantile(e,[0,.5,.8,.9,1]),'cubic res',np.mean(abs(basis(u,2)@np.linalg.lstsq(basis(u,2),v,rcond=None)[0]-v)))
  matrices[k+g]=w
p=s['panel_a']; u=p['u'];v=p['v'];f=affine(u); w=matrices['panel_av']
for _ in range(10):
 keep=np.argsort(np.linalg.norm(f@w-v,axis=1))[:39];w=np.linalg.lstsq(f[keep],v[keep],rcond=None)[0]
print('A robust',w,'range x',x.min(0),x.max(0),'y quantiles',np.quantile(y,np.linspace(0,1,11)),'state svals',np.linalg.svd(z,compute_uv=False))
Q['x']=x;Q['caldiag']=(x-w[0])/w[1:].diagonal();Q['calfull']=(x-w[0])@np.linalg.inv(w[1:]);Q['inversecal']=affine(x)@w
for b in list(Q):
 for k,W in matrices.items():
  if k=='panel_av':continue
  Q[b+k]=affine(Q[b])@W
  Q[b+k+'inv']=(Q[b]-W[0])@np.linalg.pinv(W[1:])
rng=np.random.default_rng(20); splits=[np.arange(n)]+[rng.permutation(n) for _ in range(14)]; splits=[(p[:52],p[52:]) for p in splits]
def stat(p):return (np.mean(p),p[0],max(p))
for name,f in F.items():print(name,f.shape)
a=[]
for tr,te in splits:
 d=dict(s,train={k:s['train'][k][tr] for k in ('x','y','traj')},test={'x':x[te]});p=arc.solve_batch(d);a.append((np.mean(abs(p['y_hat']-y[te])),np.mean(abs(p['state_pred']-z[te]))))
print('archive',stat(np.array(a)[:,0]),stat(np.array(a)[:,1]))
def score(K,pen):
 return stat([np.mean(abs(K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+pen*np.eye(len(tr)),y[tr],assume_a='pos')-y[te])) for tr,te in splits])
res=[]
for k,f in F.items():
 for pen in np.logspace(-6,0,13):res.append((*score(f[:n]@f[:n].T,pen),k,pen))
for k,u in Q.items():
 for mag in [0,.09,.3,1]:
  uq=u/np.sqrt(1+mag*u*u)
  for d in (2,3,4):
   f=basis(uq,d); K=f@f.T
   for pen in (1e-5,.001,.01,.1):res.append((*score(K,pen),(k,mag,d),pen))
for ranking in (lambda a:a[0],lambda a:a[1],lambda a:a[2]):
 print('best');print(*sorted(res,key=ranking)[:8],sep='\n')
