exec(open('/tmp/readout_probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares,minimize_scalar
B=s['panel_b']; kb,gb=m._fit_b(B['u'],B['v']); c=s['panel_c'];
print('panel B affine and nonlinear matrices',gb.round(3)); print('B fit RMSE',np.sqrt(np.mean((m._b_design(B['u'],kb)@gb-B['v'])**2)))
def crep(j):
 if j==1:
  def D(q,k):return np.c_[poly(q,1),np.tanh(k*q)]
  def fun(k):
   Q=np.linalg.qr(D(c['u'],k))[0]; return np.mean(((c['v1']-Q@(Q.T@c['v1']))/(1-(Q*Q).sum(1)[:,None]))**2)
  k=minimize_scalar(fun,bounds=(.3,4),method='bounded').x
 else:
  def D(q,k):return poly(q,2)
  k=0
 c0=np.linalg.lstsq(D(c['u'],k),c['v'+str(j)],rcond=None)[0]
 return D(u,k)@c0,D(z,k)@c0
for j in [1,2]:
 for i,q in enumerate(crep(j)):
  print('C transform',j,i,check(poly(q,2)),check(np.c_[poly(z,2),q]))
# Compress output to its signal-bearing components, weighting y and mean trajectory equally.
T=np.c_[Y[:,0],Y[:,1:]/np.sqrt(8)]
def candidate(q,a,b,name,seed=432):
 A=poly(q,1); D=poly(q,2); Q=np.linalg.qr(A[a])[0]; V=T[a]-Q@(Q.T@T[a]); rng=np.random.default_rng(seed)
 if name=='quartic':
  d=poly(q,4); P=np.full(d.shape[1],.12);P[:D.shape[1]]=.004
  coef=np.linalg.lstsq(np.r_[d[a],np.diag(P)],np.r_[T[a],np.zeros((len(P),T.shape[1]))],rcond=None)[0];e=abs(d[b]@coef-T[b]);return np.linalg.norm(d[a]@coef-T[a]),e[:,0].mean(),e[:,1:].mean()*np.sqrt(8),[]
 z0=np.linalg.lstsq(D[a],T[a],rcond=None)[0][:,0]
 def features(ids,p):
  if name.startswith('quot'):
   h=(A if name=='quot1' else D)[ids]@p
   return np.c_[A[ids],D[ids]/(1+h[:,None]**2)]
  if name=='gated':
   return np.c_[D[ids],(D[ids]@p[:15])*np.tanh(A[ids]@p[15:])]
  if name=='hinge':
   return np.c_[D[ids],np.maximum(D[ids]@p,0)]
  if name=='energy':
   h=D[ids]@p;return np.c_[A[ids],h,h*h]
 def res(p):
  d=features(a,p);return (d@np.linalg.lstsq(d,T[a],rcond=1e-8)[0]-T[a]).ravel()
 n={'quot1':5,'quot2':15,'gated':20,'hinge':15,'energy':15}[name]
 starts=[rng.normal(0,.4,n) for _ in range(7)]
 if name in ['hinge','energy']: starts += [z0,-z0]
 if name=='gated':starts += [np.r_[z0,rng.normal(0,.4,5)] for _ in range(3)]
 p=min((least_squares(res,p,max_nfev=110) for p in starts),key=lambda p:p.cost)
 d=features(a,p.x); assert np.isfinite(d).all(); e=abs(features(b,p.x)@np.linalg.lstsq(d,T[a],rcond=1e-8)[0]-T[b]);return np.sqrt(2*p.cost),e[:,0].mean(),e[:,1:].mean()*np.sqrt(8),p.x.round(3)
for qname,q in [('z',z),('u',u)]:
 for name in ['quot1','quot2','gated','hinge','energy','quartic']:
  print(qname,name,*candidate(q,*fs[0],name),flush=True)
