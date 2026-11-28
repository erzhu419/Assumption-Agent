import pickle,numpy as np
from itertools import combinations_with_replacement,combinations
from scipy.optimize import least_squares
from scipy.linalg import qr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; A=s['panel_a']; u=A['u']; v=A['v']; d=np.c_[np.ones(len(u)),u]; b=np.linalg.lstsq(d,v,rcond=None)[0];
for _ in range(12):
 e=np.linalg.norm(d@b-v,axis=1); w=np.minimum(1,.07/np.maximum(e,1e-12)); b=np.linalg.lstsq(d*w[:,None],v*w[:,None],rcond=None)[0]
good=np.linalg.norm(d@b-v,axis=1)<.01; b=np.linalg.lstsq(d[good],v[good],rcond=None)[0]; z=(tr['x']-b[0])@np.linalg.inv(b[1:]); p=s['panel_b']; B=np.linalg.lstsq(np.c_[np.ones(len(p['u'])),p['u']],p['v'],rcond=None)[0];
def modes(t,r):return -np.expm1(-t[:,None]*r)
def proj(h,ys):return np.linalg.lstsq(h,ys.T,rcond=None)[0].T
a=np.arange(52); q=np.arange(52,72); traj=tr['traj']; y=tr['y']; r=least_squares(lambda r:(proj(modes(t,r),traj[a])@modes(t,r).T-traj[a]).ravel(),[.4,2.],bounds=(.001,12)).x; h=modes(t,r); Y=np.c_[y,proj(h,traj)]; print('panel A inliers, residual',good.sum(),np.max(np.abs(d[good]@b-v[good]))); print('rates',r,'temporal residual',np.mean(abs(Y[:,1:]@h.T-traj),axis=0));print('B singular',np.linalg.svd(B[1:],compute_uv=False)); print('latent ranges',z.min(0),z.max(0)); print('target std',Y.std(0))
def poly(z,n):
 keys=[p for j in range(1,n+1) for p in combinations_with_replacement(range(4),j)];return np.column_stack([np.prod(z[:,p],axis=1) for p in keys]),keys
def score(p): return np.round([np.mean(abs(p[:,0]-y[q])),np.mean(abs(p[:,1:]@h.T-traj[q]))],5)
for name,Z in [('raw',tr['x']),('z',z),('Bz',z@B[1:]+B[0]),('invBz',(z-B[0])@np.linalg.inv(B[1:]))]:
 print('\n',name)
 for n in [1,2,3,4,5]:
  f,k=poly(Z,n); f=(f-f[a].mean(0))/f[a].std(0); S,U,V=None,None,None
  U,S,V=np.linalg.svd(f[a],full_matrices=False); yy=Y[a]-Y[a].mean(0);
  print('degree',n,[(l,score(Y[a].mean(0)+(f[q]@V.T*(S/(S**2+l)))@(U.T@yy)).tolist()) for l in [.0001,.01,.1,1]])
  if n!=4:continue
  for j in range(3):
   active=[]; res=yy[:,j]; out=[]
   for it in range(12):
    k0=np.argmax(abs(f[a].T@res));active.append(k0); c=np.linalg.lstsq(f[a][:,active],yy[:,j],rcond=None)[0];res=yy[:,j]-f[a][:,active]@c
    if it in [1,3,6,11]:out.append((it+1,round(np.mean(abs(Y[a,j]-Y[a,j].mean()-f[a][:,active]@c)),4),round(np.mean(abs(Y[q,j]-Y[a,j].mean()-f[q][:,active]@c)),4)))
   print('OMP',j,[k[i] for i in active],out)
