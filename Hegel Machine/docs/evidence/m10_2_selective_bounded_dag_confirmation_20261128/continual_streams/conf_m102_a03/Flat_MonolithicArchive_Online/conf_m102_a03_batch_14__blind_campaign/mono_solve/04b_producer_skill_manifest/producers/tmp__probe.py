import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
a=s['panel_a']; u,v=a['u'],a['v']; af=lambda x:np.c_[np.ones(len(x)),x]; fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
X=af(u); c=fit(X,v)
for i in range(30):
 r=np.linalg.norm(X@c-v,axis=1); w=np.sqrt(np.minimum(1,1.5*np.median(r)/np.maximum(r,1e-12))); c=fit(X*w[:,None],v*w[:,None]);
r=np.linalg.norm(X@c-v,axis=1); good=r<4*np.median(r); c=fit(X[good],v[good]); print('cal affine',c,'res sorted',np.sort(np.linalg.norm(X@c-v,axis=1)))
for k in ['panel_b','panel_c']:
 u=s[k]['u']
 for name,v in s[k].items():
  if name=='u':continue
  print(k,name)
  for n,f in [('linear',lambda u,p:u),('tanh',lambda u,p:np.tanh(p*u)),('sin',lambda u,p:np.sin(p*u)),('atan',lambda u,p:np.arctan(p*u)),('cos',lambda u,p:np.cos(p*u)),('softsign',lambda u,p:u/(1+p*abs(u))),('gaussian',lambda u,p:np.exp(-p*u*u)),('linear + tanh',lambda u,p:np.c_[u,np.tanh(p*u)]),('linear + sin',lambda u,p:np.c_[u,np.sin(p*u)]),('linear + cos',lambda u,p:np.c_[u,np.cos(p*u)])]:
   def R(p):
    X=af(f(u,p)); return (X@fit(X,v)-v).ravel()
   z=min((least_squares(R,[t],bounds=([.025],[6]),max_nfev=65) for t in [.3,1,2,4]),key=lambda z:z.cost)
   print(n,z.x,np.sqrt(np.mean(z.fun**2)))
print('time singular',np.linalg.svd(s['train']['traj'],compute_uv=False))
