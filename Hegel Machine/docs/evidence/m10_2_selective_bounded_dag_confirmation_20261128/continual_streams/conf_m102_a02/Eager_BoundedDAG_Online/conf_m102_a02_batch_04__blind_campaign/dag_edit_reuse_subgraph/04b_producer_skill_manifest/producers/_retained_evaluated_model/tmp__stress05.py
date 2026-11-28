import runpy,pickle,numpy as np
from itertools import combinations_with_replacement
m=runpy.run_path('/holdout_task/node_01.py'); cases=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('cases',len(cases)); rng=np.random.default_rng(503)
for case in cases:
 s=case['study']; t=s['train']; b=m['_calibrate'](s['panel_a']); x=m['_invert'](t['x'],b); y=np.column_stack((t['y'],t['traj'])); n,d=x.shape; near=np.roll(x,-1,axis=1); q=np.column_stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(d),2)]); a=np.c_[np.ones(n),x]; sn=np.sin(3*x); cs=np.cos(3*x); u=x-.4*sn; 
 def products(w):return np.column_stack([w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(d),2)])
 cuts=[np.arange(i,n,4) for i in range(4)]+[rng.permutation(n)[:20] for i in range(24)]+[np.arange(n-20,n)]; radial=np.linalg.norm(x,axis=1); cuts.append(np.argsort(radial)[-14:]);
 opts={'baseline':(np.c_[a,q,sn,cs],.04), 'sine-only':(np.c_[a,q,sn],.002), 'trajectory-linear-sine':(np.c_[a,sn],.001), 'warped-quadratic':(np.c_[np.ones(n),u,products(u)],.003), 'sine-plus-warped-quadratic':(np.c_[a,sn,products(u)],.005), 'raw-and-warped':(np.c_[a,sn,q,products(u)],.02), 'cyclic-mixed':(np.c_[a,q,sn,cs,np.sin(3*x*near),np.sin(3*(x+near)),np.sin(3*(x-near))],.05)}
 for name,(z,l) in opts.items():
  def fit(idx):return np.linalg.solve(z[idx].T@z[idx]+l*np.eye(z.shape[1]),z[idx].T@y[idx]);
  def err(e):return np.array([abs(e[:,0]).mean(),abs(e[:,1:]).mean()]);
  e=[err(z[v]@fit(np.setdiff1d(np.arange(n),v))-y[v]) for v in cuts]; print(name,'mean/worst/public/radial',np.round([np.mean(e[:-2],axis=0),np.max(e,axis=0),e[-2],e[-1]],5).tolist(),'train',np.round(err(z@fit(np.arange(n))-y),5));
  if name in ('sine-only','trajectory-linear-sine'): print('linear and sine',np.round(fit(np.arange(n))[np.r_[1:5,z.shape[1]-4:z.shape[1]]][:,[0,1,-1]],4))
