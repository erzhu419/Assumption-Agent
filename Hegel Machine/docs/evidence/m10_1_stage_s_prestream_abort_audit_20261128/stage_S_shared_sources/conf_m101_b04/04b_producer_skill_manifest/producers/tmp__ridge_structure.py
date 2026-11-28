from probe import *
from scipy.optimize import least_squares
from scipy.linalg import solve
from itertools import product
x=s['train']['x']; Z=s['train']['traj']; y=s['train']['y']; I=np.arange(len(x)); folds=[I[56:]]+list(np.array_split(np.random.default_rng(42).permutation(len(I)),4))+[np.argsort(np.linalg.norm(x,axis=1))[-12:]]
F=poly(x,2); Q=F[:,1:]; rng=np.random.default_rng(21)

def run(tr,te,h,tag,k,reg):
 Y=y[:,None] if h==0 else Z; M=np.eye(Y.shape[1]) if h==0 else np.linalg.svd(Y[tr],full_matrices=False)[2][:2].T; T=Y[tr]@M
 Ft=F[tr]; xt=x[tr]; basis=np.linalg.qr(Ft,mode='reduced')[0]; T0=T-basis@(basis.T@T)
 def nonlinear(x,p):
  c=x @ p[:,:4].T + p[:,4]
  if tag=='tanh':return np.tanh(c)
  if tag=='log':return np.sign(c)*np.log1p(abs(c))
  if tag=='sin':return np.sin(c)
  if tag=='rat':return c/(1+c*c)
  if tag=='qrat':return c*c/(1+c*c)
  if tag=='cubic':return c**3
 def unpack(p):return p.reshape(k,5)
 def build(p):
  H=nonlinear(xt,unpack(p)); H0=H-basis@(basis.T@H); B=solve(H0.T@H0+reg*np.eye(k),H0.T@T0,assume_a='pos'); return H,H0,B
 def res(p):
  H,H0,B=build(p); return np.r_[(H0@B-T0).ravel(),np.sqrt(reg)*B.ravel(),.002*p]
 starts=rng.normal(0,.55,(4,k*5)); sol=min((least_squares(res,p,max_nfev=95) for p in starts),key=lambda s:s.cost); H,H0,B=build(sol.x); A=fit(Ft,T-H@B)
 return (F[te]@A+nonlinear(x[te],unpack(sol.x))@B)@M.T, np.sqrt(np.mean((H0@B-T0)**2))
for h in [0,1]:
 Y=y[:,None] if h==0 else Z
 for tag,k in product(['tanh','rat','qrat','log','cubic','sin'],[1,2,3]):
  L=[run(np.setdiff1d(I,v),v,h,tag,k,1e-5 if tag in ('rat','qrat','log') else 1e-4) for v in folds]; E=np.array([np.mean(abs(P-Y[v]),axis=0) for v,(P,_) in zip(folds,L)]); print(h,tag,k,np.round([E[:-1].mean(),E[:-1].max(),E[-1].max(),np.mean([r for _,r in L])],6),flush=True)
