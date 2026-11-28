import contextlib, io, runpy
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()):
 g=runpy.run_path('/tmp/discover.py')
s=g['s']; fit=g['fit']; design=g['design']; cross=g['cross']; np.set_printoptions(precision=4,suppress=True)
p=s['panel_a']; w=g['ref'](**p); a=(s['train']['x']-w[0])/np.diag(w[1:]); p=s['panel_b']; f=lambda z:np.sign(z)*np.maximum(abs(z)-.35,0)**1.5; z=f(p['u']); wb=fit(design(z),p['v']); print('B spectrum',np.linalg.svd(wb[1:],compute_uv=False)); b=design(f(a))@wb; c=g['c1'](b)@g['wc']; d=g['c2'](b,g['r'])@g['wd']; print('C2 coeff',g['wd']); print('train/test x ranges',a.min(0),a.max(0),((s['test']['x']-w[0])/np.diag(w[1:])).min(0),((s['test']['x']-w[0])/np.diag(w[1:])).max(0))
X=design(a,c,d,cross(a,c)); names=['1']+[f'{b}{i}' for b in 'acd' for i in range(4)]+[f'a{i}c{j}' for i in range(4) for j in range(4)]; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; modes=lambda r:-np.expm1(-t[:,None]*r); splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]; rng=np.random.default_rng(87)
for ix in np.array_split(rng.permutation(len(X)),5): splits.append((np.setdiff1d(np.arange(len(X)),ix),ix))
for ix in (np.argsort(np.linalg.norm(a,axis=1))[-12:],np.argsort(np.max(abs(b),axis=1))[-12:]): splits.append((np.setdiff1d(np.arange(len(X)),ix),ix))
for si,(it,iv) in enumerate(splits):
 def residual(r):
  M=modes(r); return (M@fit(M,T[it].T)-T[it].T).ravel()
 r=least_squares(residual,[.4,1.8],bounds=([.1,1],[.9,3])).x; M=modes(r); Z=np.column_stack((y,fit(M,T.T).T)); G=X[it]; H=X[iv]; P=np.eye(X.shape[1]); P[0,0]=0
 def score(Zhat): return np.round([np.mean(abs(Zhat[:,0]-y[iv])),np.mean(abs(Zhat[:,1:]@M.T-T[iv]))],5)
 def bic_fit(G,z):
  ids=list(range(G.shape[1])); best=None
  while True:
   coef=fit(G[:,ids],z); err=np.sum((z-G[:,ids]@coef)**2); criterion=len(z)*np.log(err/len(z))+len(ids)*np.log(len(z));
   if best is None or criterion<best[0]: best=(criterion,ids[:],coef)
   if len(ids)==1: break
   candidates=[ids[:j]+ids[j+1:] for j in range(1,len(ids))]; ids=min(candidates,key=lambda ids:np.linalg.norm(z-G[:,ids]@fit(G[:,ids],z)))
  return best[1:]
 sparse=[bic_fit(G,z) for z in Z[it].T]; V=np.column_stack([H[:,j]@v for j,v in sparse]);
 print(si,'rates',np.round(r,3),'ridge',score(H@np.linalg.solve(G.T@G+.001*P,G.T@Z[it])),'sparse',score(V));
 if si<3:
  for ids,coef in sparse: print('coeff',list(zip([names[j] for j in ids],np.round(coef,3))))
