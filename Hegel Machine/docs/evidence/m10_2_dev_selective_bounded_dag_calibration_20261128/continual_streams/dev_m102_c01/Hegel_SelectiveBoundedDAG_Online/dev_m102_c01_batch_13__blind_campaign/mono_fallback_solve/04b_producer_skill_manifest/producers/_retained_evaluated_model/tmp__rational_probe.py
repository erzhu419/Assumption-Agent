exec(open('/tmp/probe.py').read().split('rec=[]')[0])
from itertools import combinations_with_replacement
from scipy.special import erf
from scipy.optimize import least_squares
from functools import partial

def poly(X,d):return np.column_stack([np.prod(X[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(4),k)])

def getpred(i,o,tar,X,d,kind,pen, numseed=3):
 F=poly(X,d); W=F[i]; yy=tar[i]; tiny=np.where(np.arange(F.shape[1])<5,.003,.006); rng=np.random.default_rng(5721)
 def design(p):
  if kind=='ratnorm':return F/(1+np.sum((X*p[:4])**2,axis=1)[:,None])
  if kind=='ratproj':return F/(1+(X@p[:4]+p[4])[:,None]**2)
  if kind=='ratexp':return F/(1+np.sum((X@p.reshape(4,4))**2,axis=1)[:,None])
  if kind=='ratmult':return F/(1+np.sum((X@p.reshape(4,2))**2,axis=1)[:,None])
  if kind=='rataff':return F/(1+np.sum(((X-p[4:])*p[:4])**2,axis=1)[:,None])
  if kind=='ratweak':return np.c_[F, X/(1+(X@p[:4]+p[4])[:,None]**2)]
  if kind=='sind': g=np.sin
  elif kind=='tanhd': g=np.tanh
  elif kind=='softd': g=lambda a:a/(1+abs(a))
  elif kind=='erb':g=erf
  elif kind=='sinp':
   return np.c_[F,np.sin(X@p[:4]),np.cos(X@p[:4])]
  elif kind=='rp':g=lambda a:a/(1+a*a)
  return np.c_[F,g(X@p[:4]+p[4])]
 dim={'ratnorm':4,'ratexp':16,'ratmult':8,'rataff':8}.get(kind,5)
 def get(p):
  M=design(p); V=M[i]; reg=np.r_[tiny,np.full(V.shape[1]-len(tiny),.008)]*pen
  c=np.linalg.solve(V.T@V+np.diag(reg*reg),V.T@yy);return M,c,reg
 def fun(p):
  M,c,reg=get(p);return np.r_[M[i]@c-yy,reg*c,.008*pen*p]
 seeds=[rng.normal(0,.3,size=dim) for _ in range(numseed)]
 if kind in ('ratnorm','rataff'):seeds[0][:4]=.45
 if kind.endswith('d') or kind in ('sinp','erb','rp','ratproj','ratweak'):seeds[0][:4]=fit(W[:,:5],yy)[1:];seeds[0][4]=0.
 opts=[least_squares(fun,p,max_nfev=100,ftol=.0001) for p in seeds]; r=min(opts,key=lambda r:r.cost);M,c,reg=get(r.x)
 return M[o]@c,np.std(M[i]@c-yy),r.x

if __name__=='__main__':
 for tar in [y,a]:
  rec=[]
  for name in ('x','z','bz','c2z','c1z'):
   X=charts[name]
   for d in (2,3):
    for kind in ('ratnorm','ratproj','ratmult','rataff','ratweak','sind','tanhd','softd','erb','sinp','rp'):
     p,e,k=getpred(*splits[0],tar,X,d,kind,1.); mae=np.mean(abs(p-tar[52:]));rec.append((mae,e,name,d,kind,k))
  for R in sorted(rec)[:18]:
   mae,e,name,d,kind,k=R
   scores=[np.mean(abs(getpred(i,o,tar,charts[name],d,kind,1.)[0]-tar[o])) for i,o in splits[1:]]
   print('struct',tar is y, R, 'oof',np.round(scores,5),flush=True)
