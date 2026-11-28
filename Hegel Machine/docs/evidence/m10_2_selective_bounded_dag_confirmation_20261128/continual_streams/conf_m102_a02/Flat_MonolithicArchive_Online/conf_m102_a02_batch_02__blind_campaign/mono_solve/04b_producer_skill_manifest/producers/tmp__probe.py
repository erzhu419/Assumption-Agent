import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; n=len(x); t=s['eval_times']
def poly(x,d):
 return np.column_stack([np.prod(x[:,p],axis=1) for k in range(d+1) for p in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def fit(F,Z):
 c=np.linalg.lstsq(F[:52],Z[:52],rcond=None)[0]; r=F@c-Z; return np.mean(abs(r[:52])),np.mean(abs(r[52:]))
ops={}
for k in 'abc':
 p=s['panel_'+k]; u=p['u']
 for b,v in p.items():
  if b=='u': continue
  f=poly(u,1); m=np.ones(len(u),bool)
  for it in range(4):
   c=np.linalg.lstsq(f[m],v[m],rcond=None)[0]; r=np.linalg.norm(f@c-v,axis=1); m=r<max(0.025,4*np.median(r))
  print(k,b,'affine',np.round(c,4), 'res',np.round(np.sort(r),5)); ops[k+b]=c
print('T svd',np.linalg.svd(T,compute_uv=False)); print('T-vs-affine SVD',np.linalg.svd(T-poly(x,1)@np.linalg.lstsq(poly(x,1),T,rcond=None)[0],compute_uv=False));print('trajectory temporal PCs',np.round(np.linalg.svd(T,full_matrices=False)[2][:4],4))
for d in range(1,5):
 f=poly(x,d); print('deg',d,fit(f,y),fit(f,T),fit(np.column_stack([f,T]),y))
for k,c in ops.items():
 z=poly(x,1)@c; print(k,'z sample',np.round(z[:3],4),'T corr',np.round(np.corrcoef(np.column_stack([z,T[:,0],T[:,-1],y]).T)[:4,4:],4))
