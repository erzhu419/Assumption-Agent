import pickle
import numpy as np
from itertools import combinations_with_replacement as cr
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']
def aff(x):return np.c_[np.ones(len(x)),x]
def poly(x,d=2):return np.c_[aff(x),*[np.prod(x[:,q],axis=1) for k in range(2,d+1) for q in cr(range(4),k)]]
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
pa=s['panel_a']; a=aff(pa['u']); v=pa['v']; ii=np.arange(len(v))
for _ in range(6):
 w=fit(a[ii],v[ii]); rr=np.linalg.norm(a@w-v,axis=1);ii=np.argsort(rr)[:-3]
print('A affine',w,'scale gram',w[1:]@w[1:].T,'tail residuals',np.sort(rr)[-7:]);print('test radii',np.quantile(np.linalg.norm(s['test']['x'],axis=1),[0,.5,.9,1]))
F=poly(x);G=poly(x,3);E=-np.expm1(-t[:,None]*[1.52,.22]); Z=z@np.linalg.pinv(E).T; R=np.sum(x*x,axis=1)
splits=list(np.array_split(np.arange(len(x)),6))+[np.argsort(R)[-12:]]
ys={}; zs={}
for j in splits:
 i=np.setdiff1d(np.arange(len(x)),j); xa=x[i]; ya=y[i]; f=F[i]; g=G[i]; r=R[i]; p=fit(f,ya);h=np.zeros((4,4))
 for (a,b),c in zip(cr(range(4),2),p[5:]):h[a,b]=h[b,a]=c*(1 if a==b else .5)
 val,vec=np.linalg.eigh(h);v=vec[:,-1]; lin=p[1:5]
 def add(name,pred):ys.setdefault(name,[]).append(np.mean(abs(y[j]-pred)))
 for name,fn in {'sin':np.sin,'tanh':np.tanh,'asinh':np.arcsinh,'rat':lambda u:u/(1+u*u),'cubic':lambda u:u**3,'softabs':lambda u:u/(1+abs(u)),'exp':lambda u:np.exp(np.clip(u,-10,8))}.items():
  for typ in ('radial','quadratic','mixed'):
   def features(q,xx,ff,rr):
    u=q[0]+xx@q[1:5]
    if typ=='mixed':u=u+q[5]*rr
    return np.c_[ff if typ=='quadratic' else np.c_[aff(xx),rr],fn(u)]
   starts=[np.r_[bias,d*scale,0.] for d in (v,lin/np.linalg.norm(lin)) for scale in (.5,1.4) for bias in (0.,.6)]
   def residual(q):
    h=features(q,xa,f,r); return h@fit(h,ya)-ya
   sol=min((least_squares(residual,q,bounds=(-5,5),max_nfev=85) for q in starts),key=lambda sol:sol.cost)
   h=features(sol.x,xa,f,r);add((name,typ),features(sol.x,x[j],F[j],R[j])@fit(h,ya))
 for typ in ('tanh','rad-den','link-den','link-den-quad','logistic','sinh','softabs'):
  def fn(q,xx,ff,rr):
   base=ff@q[:15]; sc=q[15]
   if typ=='tanh':return (np.tanh(sc*base)/sc if sc>1e-8 else base)
   if typ=='sinh':return (np.sinh(np.clip(sc*base,-6,6))/sc if sc>1e-8 else base)
   if typ=='logistic':return q[16]+(np.tanh(sc*base)/sc if sc>1e-8 else base)
   if typ=='rad-den':return base/(1+sc*rr)
   if typ=='softabs':return base/(1+sc*abs(base))
   den=q[16]+xx@q[17:21]; num=base
   if typ=='link-den-quad':den=den+sc*rr
   return num/(1+den*den)
  starts=[np.r_[p,.2,0.,scale*d] for d in (v,lin/np.linalg.norm(lin)) for scale in (.15,.55)]
  sol=min((least_squares(lambda q:fn(q,xa,f,r)-ya,q,bounds=(np.r_[np.repeat(-10.,15),0.,np.repeat(-5.,5)],np.repeat(10.,21)),max_nfev=100) for q in starts),key=lambda sol:sol.cost)
  add(typ,fn(sol.x,x[j],F[j],R[j]))
 # Time coefficient regularization and supervised response features, using only fitted scalar values at validation time.
 D=cdist(x,x,'sqeuclidean'); k=(1+np.sqrt(5*D)/3+5*D/27)*np.exp(-np.sqrt(5*D)/3)+F@F.T
 yp=k[np.ix_(j,i)]@np.linalg.solve(k[np.ix_(i,i)]+.0002*np.eye(len(i)),ya)
 for weights in [(1,0),(1,.05),(1,.3),(1,1),(1,3)]:
  for extra in (False,True):
   dev=np.c_[F[i],ya,xa*ya[:,None]] if extra else F[i]; test=np.c_[F[j],yp,x[j]*yp[:,None]] if extra else F[j]
   h=dev@dev.T+weights[1]*(k-F@F.T)[np.ix_(i,i)]; hh=test@dev.T+weights[1]*(k-F@F.T)[np.ix_(j,i)]; pred=(hh@np.linalg.solve(h+(.00003 if weights[1] else .001)*np.eye(len(i)),Z[i]))@E.T
   err=abs(z[j]-pred);zs.setdefault((weights[1],extra),[]).append([err.mean(),err[:,-1].mean(),err.mean(0).max()])
 for length,noise in [(2,.0001),(1.4,.001)]:
  k=F@F.T+np.exp(-D/(2*length**2));add(('rbf',length,noise), k[np.ix_(j,i)]@np.linalg.solve(k[np.ix_(i,i)]+noise*np.eye(len(i)),ya))
for name,e in sorted(ys.items(),key=lambda r:np.mean(r[1][:-1])+.7*max(r[1]))[:20]:print('y',name,np.round(e,5),np.mean(e[:-1]),max(e))
for name,e in zs.items():
 a=np.array(e);print('traj',name,'average',a[:-1].mean(0).round(5),'worst',a.max(0).round(5),'radius',a[-1].round(5))
