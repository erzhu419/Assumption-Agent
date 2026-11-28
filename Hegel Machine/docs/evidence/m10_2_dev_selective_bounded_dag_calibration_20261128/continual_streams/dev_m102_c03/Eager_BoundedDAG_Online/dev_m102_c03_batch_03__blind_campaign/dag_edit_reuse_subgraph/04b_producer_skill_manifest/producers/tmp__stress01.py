import numpy as np, pickle, runpy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=len(x)
b=runpy.run_path('/tmp/node01_baseline.py'); c=runpy.run_path('/holdout_task/node_01.py');
z=c['_latent'](x,s['panel_a']); groups={}
rng=np.random.default_rng(56)
groups['fresh random']=[rng.permutation(n)[-20:] for _ in range(24)]
groups['coordinate tails']=[np.argsort(sign*z[:,a])[-12:] for a in range(4) for sign in (-1,1)]
groups['largest radius']=[np.argsort(np.linalg.norm(z,axis=1))[-14:]]
for name,S in groups.items():
 for tag,m in [('baseline',b),('challenger',c)]:
  E=[]
  for j in S:
   i=np.setdiff1d(np.arange(n),j); study=dict(s,train={k:tr[k][i] for k in ('x','y','traj')},test={'x':x[j]})
   p,t=m['_predict'](study); assert p.shape==tr['y'][j].shape and t.shape==tr['traj'][j].shape and np.isfinite(p).all() and np.isfinite(t).all()
   E.append([np.mean(abs(p-tr['y'][j])),np.mean(abs(t-tr['traj'][j]))])
  print(name,tag,'mean',np.mean(E,0),'max',np.max(E,0))
u,v=[s['panel_a'][k] for k in ('u','v')]; h=np.c_[np.ones(len(u)),u]; keep=np.linalg.norm(v-h@c['_fit'](h,v),axis=1)<2
print('Cubic inverse panel inlier recovery median/max',np.quantile(np.linalg.norm(c['_latent'](v[keep],s['panel_a'])-u[keep],axis=1),[.5,1]))
