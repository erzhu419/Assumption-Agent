exec(open('/tmp/operators.py').read().split('res=[]')[0])
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
z=B(A(x)); z=(z-z.mean(0))/z.std(0)
Yt=Y[:,1:]; modes=np.linalg.svd(Yt,full_matrices=False)[2][:2]; YY=np.c_[Y[:,0],Yt@modes.T]
rng=np.random.default_rng(228); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in (rng.permutation(72) for _ in range(8))]+[(p[:52],p[52:]) for p in (np.random.default_rng(1983).permutation(72) for _ in range(1))]

def fitK(z,a,b,tar,kind,poly,ard):
 d=(z[:,None]-z[None,:])**2; q=(z@z.T/4+1)**2; p=aff(z); n=len(a); aa=np.ix_(a,a); ba=np.ix_(b,a)
 def mats(h):
  g=np.exp(h[:4] if ard else np.repeat(h[0],4)); R=d@g
  if kind=='gauss': k=np.exp(-R); kd=-k
  else:
   r=np.sqrt(5*R); k=(1+r+r**2/3)*np.exp(-r); kd=-5/6*(1+r)*np.exp(-r)
  sg=np.exp(h[-1]); k+=poly*q
  return k,kd,g,sg
 def obj(h,ret=False):
  k,kd,g,sg=mats(h); L=cho_factor(k[aa]+sg*np.eye(n)); pa=p[a]; ap=cho_solve(L,pa); pp=cho_factor(pa.T@ap); v=cho_solve(L,tar[a]); c=cho_solve(pp,pa.T@v); v-=ap@c
  if ret:return p[b]@c+k[ba]@v
  rss=np.sum(tar[a]*v); dof=n-pa.shape[1]; loss=dof*np.log(rss/dof)+2*np.log(L[0].diagonal()).sum()+2*np.log(pp[0].diagonal()).sum()+.35*np.sum((h[:(-1)]-np.log(.13))**2)
  H=cho_solve(L,np.eye(n))-ap@cho_solve(pp,ap.T); M=H-dof/rss*np.outer(v,v); gk=np.einsum('ij,ijk,ij,k->k',M,d[aa],kd[aa],g); return loss,np.r_[gk if ard else gk.sum(),sg*np.trace(M)]+np.r_[.7*(h[:-1]-np.log(.13)),0]
 init=np.log([.13]*(4 if ard else 1)+[.003]); r=minimize(obj,init,jac=True,bounds=[(-5,1.5)]*(len(init)-1)+[(-12,-2)],method='L-BFGS-B');return obj(r.x,True),np.exp(r.x)
for name,w in [('AB',z),('raw',(x-x.mean(0))/x.std(0))]:
 for kind in ['gauss','matern']:
  for poly in [.0,.05,.3]:
   for ard in [False,True]:
    E=[]; E2=[]
    for a,b in splits:
     yp,h=fitK(w,a,b,YY[:,0],kind,poly,ard); E.append(abs(yp-YY[b,0]).mean());
     _,_,mh=np.linalg.svd(Yt[a],full_matrices=False); mh=mh[:2]; cp=Yt@mh.T; tp=np.column_stack([fitK(w,a,b,cp[:,i],kind,poly,ard)[0] for i in range(2)])@mh; E2.append(abs(tp-Yt[b]).mean());
     if len(E)==1: h0=h
    print(name,kind,poly,ard,np.round([E[0],np.mean(E),max(E),E2[0],np.mean(E2),max(E2)],5),np.round(h0,5),flush=True)
