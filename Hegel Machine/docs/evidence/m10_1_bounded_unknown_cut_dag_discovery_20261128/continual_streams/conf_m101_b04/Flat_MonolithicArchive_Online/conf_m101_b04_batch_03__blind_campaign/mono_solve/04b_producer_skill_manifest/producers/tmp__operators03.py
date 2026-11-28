import numpy as np,itertools,pickle
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x):return np.c_[np.ones(len(x)),x]
def quad(x):return np.c_[aff(x),*[x[:,j]*x[:,k] for j,k in itertools.combinations_with_replacement(range(4),2)]]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def proj(u,v,fn,base,share):
 def des(x,k):return np.c_[base(x),fn(x*k)]
 def r(k):
  a=des(u,k);return (a@fit(a,v)-v).ravel()
 out=min((least_squares(r,[k]*(1 if share else 4),bounds=(.25,5.5),max_nfev=140) for k in (0.7,1.3,2,3,4.7)),key=lambda z:np.mean(z.fun**2))
 a=des(u,out.x);w=fit(a,v);h=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,1);e=(a@w-v)/(1-h[:,None]);return np.mean(e**2)**.5,out.x,w,lambda x:des(x,out.x)@w

def panels(verbose=False):
 u,v=(S['panel_a'][k] for k in ('u','v')); keep=np.arange(len(u))
 for it in range(4):
  d=np.array([fit(aff(u[keep,j:j+1]),v[keep,j]) for j in range(4)]).T
  e=np.linalg.norm(d[0]+u*d[1]-v,axis=1);keep=np.argsort(e)[:38] if it==0 else np.flatnonzero(e<8*np.median(e))
 ops=[lambda x:d[0]+x*d[1]]
 for pn,j,fns,bases in [('panel_b','v',[np.sin,np.tanh],[aff]),('panel_c','v1',[np.sin,np.tanh],[aff]),('panel_c','v2',[np.cos],[aff,quad])]:
  u,v=S[pn]['u'],S[pn][j]; fits=[]
  for fn,base,share in itertools.product(fns,bases,[True,False]):
   p=proj(u,v,fn,base,share);fits.append(p)
   if verbose:print(pn,j,fn.__name__,base.__name__,share,round(p[0],7),np.round(p[1],5))
  p=min(fits,key=lambda z:z[0]);ops.append(p[-1])
  if verbose: print('best coeffs',np.round(p[2],4))
 return ops

def relax(tr,t):
 def basis(r):return -np.expm1(-t[:,None]*r)
 def r(k):
  a=basis(k);return (a@fit(a,tr.T)-tr.T).ravel()
 rr=least_squares(r,[1.8,.41],bounds=([.7,.05],[4,.65])).x
 B=np.linalg.qr(basis(rr))[0];return tr@B,B

def coords(x):
 A,B,C,D=panels();ops=[B,C,D]; names=['B','C','D']; zz={'X':x,'A':A(x)}
 for src in ['X','A']:
  for n,f in zip(names,ops):
   nt=src+n;zz[nt]=f(zz[src])
   for m,g in zip(names,ops):zz[nt+m]=g(zz[nt]);
 return zz

def gp(x,xt,p,pt,y,alpha=.002,l=.3):
 k=np.exp(-l*cdist(x,x,'sqeuclidean'));kt=np.exp(-l*cdist(xt,x,'sqeuclidean'));cf=cho_factor(k+alpha*np.eye(len(k)));v=cho_solve(cf,y);a=cho_solve(cf,p);w=fit(p.T@a,p.T@v);return pt@w+kt@(v-a@w)
if __name__=='__main__':
 panels(True)
 x=S['train']['x'];y=S['train']['y'];tr=S['train']['traj'];t=S['eval_times']; zz=coords(x); rng=np.random.default_rng(43); splits=[np.arange(72),*[rng.permutation(72) for _ in range(5)]]; scores=[]
 for name,z in zz.items():
  for typ in ('aff','quad','gpraw','gp'):
   rec=[]
   for ix in splits:
    a,b=ix[:52],ix[52:];Y,B=relax(tr[a],t); target=np.c_[y[a],Y]; p=(aff if typ=='aff' else quad)(z)
    if typ.startswith('gp'):
     xx=x if typ=='gpraw' else z;pred=gp(xx[a],xx[b],p[a],p[b],target)
    else: pred=p[b]@fit(p[a],target)
    rec.append([np.abs(pred[:,0]-y[b]).mean(),np.abs(pred[:,1:]@B.T-tr[b]).mean()])
   scores.append((name,typ,np.array(rec)))
 for col in (0,1):
  print('leaderboard',col)
  for nm,ty,a in sorted(scores,key=lambda o:o[2][:,col].mean())[:14]: print(nm,ty,np.round(a[:,col],4),round(a[:,col].mean(),4))
