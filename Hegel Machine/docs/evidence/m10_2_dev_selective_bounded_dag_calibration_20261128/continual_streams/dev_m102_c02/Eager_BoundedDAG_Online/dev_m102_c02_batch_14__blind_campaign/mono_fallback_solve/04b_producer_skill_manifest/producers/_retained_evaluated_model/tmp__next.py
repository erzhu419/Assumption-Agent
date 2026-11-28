import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); from batch_solver import _build,_ls,_aff,_outer,solve_batch
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge, Lasso
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; n=52; y=T['y']; t=T['traj']; V=np.c_[y,t]; z,b,D=_build(s,T['x']); i,j=np.triu_indices(4); q=b[:,i]*b[:,j]; cross=_outer(b,z)
np.set_printoptions(precision=4,suppress=True)
U,S,H=np.linalg.svd(t[:n],full_matrices=False); Y=np.c_[y,(t@H[:3].T)/S[:3]*np.sqrt(n)]; R=_ls(D[:n],Y[:n]); print('coef',R); print('outer svd',*[np.linalg.svd(R[-16:,k].reshape(4,4),compute_uv=False) for k in range(4)])
print('y residual vs y',np.corrcoef(np.c_[y,t@H[:3].T,y-D@R[:,0]].T)[-1]); print('times coeff',H[:3]);
A=np.c_[_aff(b),q,cross]; p=_ls(A[:n],V[:n]); E=A@p-V; print('base',np.mean(abs(E[n:]),axis=0))
# Readout libraries built from the recovered stage coordinates and latent time amplitudes.
v=A@_ls(A[:n],t[:n]@H[:3].T); vv=v/np.std(v[:n],0); h=z-.39752*np.sin(2.8478*z-.002); B=[A,np.c_[A,z],np.c_[A,(z[:,i]*z[:,j])],np.c_[A,z,z[:,i]*z[:,j]],np.c_[A,_outer(b,b*b),_outer(z,b*b)],np.c_[A,_outer(b,z*z),_outer(b,np.cos(2.15*z))],np.c_[A,vv*vv,_outer(vv,b),_outer(vv,z)],np.c_[A,_outer(b,h*h),_outer(h,z*z)]]
for k,a in enumerate(B):
 scale=np.maximum(a[:n].std(0),.01); a=a/scale
 row=[]
 for alpha in [0,.001,.01,.05,.2,1,4]:
  r=Ridge(alpha=alpha).fit(a[:n,1:],y[:n]); P=r.predict(a[:,1:]); row.append((round(np.mean(abs(P[n:]-y[n:])),4),round(np.mean(abs(P[:n]-y[:n])),4),alpha))
 for alpha in [.0003,.001,.003,.01,.03]:
  r=Lasso(alpha=alpha,max_iter=30000).fit(a[:n,1:],y[:n]); P=r.predict(a[:,1:]); row.append((round(np.mean(abs(P[n:]-y[n:])),4),round(np.mean(abs(P[:n]-y[:n])),4),-alpha))
 print('lib',k,sorted(row)[:4]);
P=s['panel_b']; u,w=P['u'],P['v']; g=lambda u:u-.39752*np.sin(2.8478*u-.002); base=_aff(g(u)); r=w-base@_ls(base,w); print('operator res',np.mean(abs(r)),w[:3]);
bs=[]
for label,d in [('sq',u*u),('sin',np.sin(2.85*u)),('cos',np.cos(2.85*u)),('cub',u**3),('prod',u[:,i]*u[:,j]),('sat',np.tanh(w)),('out2',w*w),('out3',w**3),('sinout',np.sin(2.15*w)),('mixprod',_outer(u,w))]:
 a=np.c_[base,d]; res=a@_ls(a,w)-w; print('op',label,np.mean(abs(res)),np.linalg.svd(_ls(a,w)[5:],compute_uv=False))
# Model potential scalar readout ratios with a small shared nonlinear gate.
for name,G in [('b',b),('z',z),('amp',vv),('bz',np.c_[b,z])]:
 G=G/G[:n].std(0)
 for kind in ('ratio','sin','tanh','sinadd','tanhadd'):
  out=[]
  for start in np.r_[np.zeros((1,G.shape[1])),np.eye(G.shape[1]),np.random.default_rng(1).normal(0,.5,(8,G.shape[1]))]:
   def design(p):
    r=p[0]+G@p[1:];
    if kind=='ratio':return A/(1+r*r)[:,None]
    if kind.endswith('add'):return np.c_[A,(np.sin if kind=='sinadd' else np.tanh)(r)]
    return A*(1+.22*(np.sin if kind=='sin' else np.tanh)(r))[:,None]
   def ff(p):
    a=design(p)[:n];return np.r_[a@_ls(a,y[:n])-y[:n],.008*p]
   o=least_squares(ff,np.r_[.1,start],max_nfev=120); a=design(o.x);e=abs(a@_ls(a[:n],y[:n])-y);out.append((e[:n].mean(),e[n:].mean(),o.x))
  print('gate',name,kind, sorted(out,key=lambda a:a[0])[:1], min(a[1] for a in out),flush=True)
