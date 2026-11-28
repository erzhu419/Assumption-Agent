import pickle,numpy as np,itertools,runpy
np.set_printoptions(precision=4,suppress=True,linewidth=145)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); a=f['_calibration'](s['panel_a']); x=np.linalg.solve(a[1:].T,(s['train']['x']-a[0]).T).T; y=s['train']['y']; t=s['train']['traj']; print('calibration',a,'x ranges',x.min(0),x.max(0)); print('traj svd centered',np.linalg.svd(t-t.mean(0),compute_uv=False)); print('mean/range',t.mean(0),t.min(0),t.max(0)); print('temporal SVD',np.linalg.svd(t,full_matrices=False)[2][:5]);
def aff(u):return np.c_[np.ones(len(u)),u]
def poly(u,d):
 return np.column_stack([np.prod(u[:,ii],axis=1) for k in range(d+1) for ii in itertools.combinations_with_replacement(range(u.shape[1]),k)])
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; print(k)
 for ky,v in p.items():
  if ky=='u':continue
  print(ky,'range',v.min(0),v.max(0),'affcoef',np.linalg.lstsq(aff(u),v,rcond=None)[0]);
  for label,b in [('aff',aff(u)),('quad',poly(u,2)),('trig',np.c_[aff(u),np.sin(u),np.cos(u)]),('relu',np.c_[aff(u),np.abs(u)])]:
   w=np.linalg.lstsq(b,v,rcond=None)[0]; r=v-b@w; leverage=np.sum(np.linalg.svd(b,full_matrices=False)[0]**2,1); print(label,'med/mean abs',np.median(abs(r),0),np.mean(abs(r),0),'loo',np.mean(abs(r/np.maximum(1e-9,1-leverage[:,None])),0));
  if k!='panel_a':print(np.c_[u,v]);
rng=np.random.default_rng(42); folds=np.array_split(rng.permutation(len(x)),6)
for d in (1,2,3,4):
 b=poly(x,d); yy=np.c_[y,t]; err=[]
 for q in folds+[np.arange(52,72)]:
  p=np.setdiff1d(np.arange(len(x)),q); w=np.linalg.lstsq(b[p],yy[p],rcond=None)[0];err.append(np.mean(abs(yy[q]-b[q]@w),0))
 print('polynomial',d,np.asarray(err))
