import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
poly=lambda x,d: np.column_stack([np.prod(x[:,c],axis=1) if c else np.ones(len(x)) for k in range(d+1) for c in it.combinations_with_replacement(range(x.shape[1]),k)])
u,v=(s['panel_a'][k] for k in ['u','v']); a=aff(u); e=np.linalg.norm(a@ls(a,v)-v,axis=1); m=e<4*np.median(e); u,v=u[m],v[m]; a=aff(u)
for tag,f in {'aff':a,'sq':np.c_[a,u*u],'abs':np.c_[a,np.abs(u)],'cubic':np.c_[a,u**3],'poly2':poly(u,2),'poly3':poly(u,3),'radial':np.c_[a,u*(u*u).sum(1)[:,None]],'oddpoly2':np.c_[a,u*np.abs(u)],'sin':np.c_[a,np.sin(u)],'tanh':np.c_[a,np.tanh(u)]}.items():
 w=ls(f,v); loo=(f@w-v)/np.maximum(1e-12,1-np.einsum('ij,ji->i',f,np.linalg.pinv(f)))[:,None]; print(tag,np.sqrt(np.mean((f@w-v)**2)),np.mean(np.abs(loo))); print(w.T if tag in ['sq','cubic','aff','poly2'] else '')
for j in [1,2]:
 u=s['panel_c']['u']; v=s['panel_c']['v'+str(j)]
 def des(x,k): return np.c_[aff(x), x/(1+k*abs(x)) if j==1 else x*x/(1+k*x*x)]
 def r(k):
  q=des(u,k[0]);return (q@ls(q,v)-v).ravel()
 k=least_squares(r,[.5],bounds=(.001,10)).x;print('C',j,k,np.std(r(k)))
x=s['train']['x']; z=np.c_[s['train']['y'],s['train']['traj']];
for d in range(1,5):
 f=poly(x,d);e=abs(f[52:]@ls(f[:52],z[:52])-z[52:]);print('raw polynomial',d,e[:,0].mean(),e[:,1:].mean())
