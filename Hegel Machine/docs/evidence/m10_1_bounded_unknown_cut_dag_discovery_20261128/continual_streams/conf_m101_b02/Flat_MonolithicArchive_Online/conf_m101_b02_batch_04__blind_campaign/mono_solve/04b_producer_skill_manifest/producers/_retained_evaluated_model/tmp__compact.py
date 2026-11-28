import contextlib,io,runpy
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): g=runpy.run_path('/tmp/discover.py')
s=g['s']; fit=g['fit']; design=g['design']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; n=len(y)
w=g['ref'](**s['panel_a']); a=(x-w[0])/np.diag(w[1:]); p=s['panel_b']
shrink=lambda z:np.sign(z)*np.maximum(abs(z)-.35,0)**1.5
z=shrink(p['u']); v=p['v']; zc=z-z.mean(0); vc=v-v.mean(0); U,S,V=np.linalg.svd(zc.T@vc); W=(U@V)*(S.sum()/np.sum(zc**2)); b=(shrink(a)-z.mean(0))@W+v.mean(0)
c=g['c1'](b)@g['wc']; d=g['c2'](b,g['r'])@g['wd']; dot=np.sum(a*c,axis=1); asym=a[:,1]*c[:,0]; diag=a*c
X=design(a,c,d,g['cross'](a,c)); P=np.eye(X.shape[1]);P[0,0]=0
S0=design(a,c,dot); S1=design(a,c,dot,asym); S2=design(a,c,diag); F0=design(a,asym); F1=design(a,asym,c); L0=design(c,dot); L1=design(a,c,dot); L2=design(c,dot,asym)
print('B constrained panel residual',np.linalg.norm(zc@W-vc))
splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]; rng=np.random.default_rng(87)
for iv in [*np.array_split(rng.permutation(n),5), np.argsort(np.linalg.norm(a,axis=1))[-12:], np.argsort(np.max(abs(b),axis=1))[-12:]]: splits.append((np.setdiff1d(np.arange(n),iv),iv))
for k,(it,iv) in enumerate(splits):
 print('slice',k,'y',*[round(np.mean(abs(D[iv]@fit(D[it],y[it])-y[iv])),5) for D in [S0,S1,S2]])
 for label,F,L in [('compact',F0,L0),('slow_a',F0,L1),('slow_asym',F0,L2),('fast_c',F1,L0),('both_a_c',F1,L1),('ridge',X,X)]:
  def G(ix,r):
   m=-np.expm1(-t[:,None]*r); return np.concatenate((F[ix,None,:]*m[None,:,0,None],L[ix,None,:]*m[None,:,1,None]),axis=-1).reshape(len(ix)*len(t),-1)
  Y=T[it].ravel()
  def beta(A): return np.linalg.solve(A.T@A+np.kron(np.eye(2),P)*.001,A.T@Y) if label=='ridge' else fit(A,Y)
  def res(r):
   A=G(it,r); return A@beta(A)-Y
  r=least_squares(res,[1.8,.4],bounds=([1,.1],[3,.9])).x
  pred=(G(iv,r)@beta(G(it,r))).reshape(len(iv),-1); err=np.abs(pred-T[iv]); print(label,np.round(r,4),round(np.std(res(r)),5),round(err.mean(),5),round(err.mean(0).max(),5))
  if k==0 and label in ('compact','fast_c','slow_a'): print('weights',np.round(beta(G(it,r)),3))
