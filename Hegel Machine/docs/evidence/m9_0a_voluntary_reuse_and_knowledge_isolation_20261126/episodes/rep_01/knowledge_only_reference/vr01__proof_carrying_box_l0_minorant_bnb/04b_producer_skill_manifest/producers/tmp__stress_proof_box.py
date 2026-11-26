from check_proof_box import check,m
import numpy as np,time

def case(name,X,y,l0,l2,M,expected=None):
 q=dict(X=X,y=y,lam0=l0,lam2=l2,M=M); t=time.monotonic()
 try:
  s=m.solve_proof_carrying_box_l0(q); stats=check(q,s)
  if expected is not None:
   assert abs(s['objective']-expected)<=1e-5, (s['objective'],expected)
  print(name,stats,'objective',s['objective'],'seconds',round(time.monotonic()-t,3),flush=True)
 except Exception as e:
  print(name,type(e).__name__,e,'seconds',round(time.monotonic()-t,3),flush=True)

p=22; l2=.14; l0=.22
for M in (.29,1.1):
 y=np.r_[np.linspace(-2,-.4,11),np.full(11,np.sqrt(2*l0*(1+l2)))]; b=np.minimum(abs(y)/(1+l2),M)
 case(f'orthogonal with ties and box {M}',np.eye(p),y,l0,l2,M,.5*(y@y)+np.minimum(0,l0-b*(abs(y)-.5*(1+l2)*b)).sum())
for seed,rho in ((0,.26),(1,.26),(2,.42),(3,-.42)):
 rng=np.random.default_rng(seed); G=np.eye(p)+np.diag(np.full(p-1,rho),1)+np.diag(np.full(p-1,rho),-1); X=np.linalg.cholesky(G).T
 y=np.linalg.solve(X.T,np.sqrt(2*l0*(1+l2))*rng.uniform(.87,1.13,p)*rng.choice([-1,1],p))
 case(f'coupled path {seed} {rho}',X,y,l0,l2,1.1)
for seed,n,l0 in ((5,35,.16),(6,28,.23),(7,15,.26),(8,35,.16)):
 rng=np.random.default_rng(seed); X=rng.normal(size=(n,p)); X/=np.linalg.norm(X,axis=0); w=np.zeros(p); w[rng.choice(p,5,replace=False)]=rng.uniform(.55,1.2,5)*rng.choice([-1,1],5)
 case(f'dense planted {seed} n={n}',X,X@w+.16*rng.normal(size=n),l0,l2,.8)
for pairs in (2,5,8):
 cols=np.arange(p); rows=cols.copy(); rows[1:2*pairs:2]-=1; X=np.eye(p)[:,rows]; y=np.zeros(p); y[:2*pairs:2]=1
 case(f'{pairs} independent duplicate ties',X,y,.15,.12,1.,pairs*(.5+.15-.5/1.12))
