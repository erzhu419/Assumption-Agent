exec(open('/tmp/probe.py').read().split('for name in')[0])
from scipy.optimize import least_squares
from scipy.linalg import subspace_angles
from scipy.interpolate import RBFInterpolator
u,v=s['panel_a'].values(); P=pol(u,3); names=['1']+[''.join(map(str,j)) for k in [1,2,3] for j in C(range(4),k)]; b=robust(P,v,50); err=np.linalg.norm(P@b-v,axis=1); keep=np.argsort(err)[:40]; print('cal err',np.sort(err));
for j in range(4):
 print('cal',j,sorted(zip(names,b[:,j].round(5)),key=lambda p:-abs(p[1]))[:15]); idx=np.where(abs(b[:,j])>.004)[0]; bj=np.linalg.lstsq(P[keep][:,idx],v[keep,j],rcond=None)[0]; b[:,j]=0; b[idx,j]=bj; print('sparse',list(zip(np.array(names)[idx],bj.round(6))),np.sort(abs(P@b[:,j]-v[:,j]))[39]);
for name,key in [('panel_c','v2'),('panel_c','v1'),('panel_b','v')]:
 p=s[name]; pp=pol(p['u'],2); bb=np.linalg.lstsq(pp,p[key],rcond=None)[0]; print(name,key,'maxerr',abs(pp@bb-p[key]).max()); print(np.column_stack((names[:15],bb.round(4)))); print('singular',np.linalg.svd(bb[1:5],compute_uv=False))
a=pol(u,1); print('u A range',u.min(0),u.max(0)); X=s['train']['x']; xi=(X-b[0])@np.linalg.pinv(b[1:5]);
for k in range(25):
 xi-=(pol(xi,3)@b-X)@np.linalg.pinv(b[1:5])*.5
print('inverse residual',abs(pol(xi,3)@b-X).max(), 'range',xi.min(0),xi.max(0));
z=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; U,S,V=np.linalg.svd(z,full_matrices=False); print('time basis',V[:2]);
for name,f in [('2exp',lambda p:np.array([-np.expm1(-p[0]*t),-np.expm1(-p[1]*t)])),('exp+linear',lambda p: np.array([-np.expm1(-p[0]*t),t])),('exp+expt',lambda p: np.array([-np.expm1(-p[0]*t),t*np.exp(-p[1]*t)])),('exp+tanh',lambda p: np.array([-np.expm1(-p[0]*t),np.tanh(p[1]*t)]))]:
 def cost(p):
  F=f(p); return (z-z@np.linalg.pinv(F)@F).ravel()
 res=least_squares(cost,[1.3,.3][:1 if name=='exp+linear' else 2],bounds=(.002,8)); print('time',name,res.x,np.sqrt(np.mean(res.fun**2)))
Z=np.column_stack((y,z));
for name,q in [('raw',X),('inv',xi),('inv-squash',xi/np.sqrt(1+.09*xi**2))]:
 for d in range(1,5):
  F=pol(q,d); err=[]
  for i,j in [(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),k),k) for k in np.array_split(np.random.default_rng(23).permutation(72),4)]:
   for lam in [.003]:
    pred=F[j]@np.linalg.solve(F[i].T@F[i]+lam*np.eye(F.shape[1]),F[i].T@Z[i]); err.append([abs(pred[:,0]-y[j]).mean(),abs(pred[:,1:]-z[j]).mean()]);
  print('cv',name,d,np.array(err).round(4).tolist())
np.savez('/tmp/inverse_probe.npz',coef=b,latent=xi)
