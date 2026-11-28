from explore import *
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
np.set_printoptions(precision=4,suppress=True,linewidth=140)

def base(q): return np.c_[np.ones(len(q)),q]
def quad(q): return PolynomialFeatures(2).fit_transform(q)
def loo(f,v):
 h=f@np.linalg.pinv(f); return np.sqrt(np.mean(((h@v-v)/np.maximum(1-h.diagonal(),1e-9)[:,None])**2))
u=pc['u']; specs={}
for n in ('v1','v2'):
 v=pc[n]; fits=[]
 for mode,g in [('tanh',np.tanh),('sin',np.sin),('rat',lambda x:x/(1+x*x)),('cos',np.cos),('gauss',lambda x:np.exp(-x*x))]:
  for trend in ('linear','diag','all'):
   def f(q,k,trend=trend,g=g):
    p=base(q) if trend=='linear' else np.c_[base(q),q*q] if trend=='diag' else quad(q)
    return np.c_[p,g(k*q)]
   for k in np.r_[np.arange(.5,4.6,.2),1.7,2.0]:
    fits.append((loo(f(u,k),v),mode,trend,k,f))
 fits.sort(key=lambda t:t[0]); print(n,[(round(a,6),b,c,round(d,2)) for a,b,c,d,_ in fits[:8]])
 for i in range(3):
  _,m,tr,k,f=fits[i]; a=f(u,k); b=np.linalg.lstsq(a,v,rcond=None)[0]; print(n,i,'coef',b.T)
 _,m,tr,k,f=fits[0]; specs[n]=(f,k,np.linalg.lstsq(f(u,k),v,rcond=None)[0])

def c(q,n):
 f,k,b=specs[n]; return f(q,k)@b

if __name__=='__main__':
 ns=len(x); folds=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(26).permutation(72),4)]
 coords2=dict(coords)
 for k in ('x','A','Ai','B','AB','AiB'):
  for v in ('v1','v2'):
   q=coords[k]; coords2[k+v]=c(q,v); coords2[k+v+'B']=tf(c(q,v),B)
 def score(f,alpha):
  vals=[]
  for tr,te in folds:
   mu=f[tr].mean(0); sd=f[tr].std(0); use=sd>1e-8; a=(f[:,use]-mu[use])/sd[use]; b=np.linalg.solve(a[tr].T@a[tr]+alpha*np.eye(a.shape[1]),a[tr].T@target[tr]); vals.append(np.abs(a[te]@b+target[tr].mean(0)-target[te]).mean(0))
  e=np.array(vals); return tuple(np.round([e[:,0].mean(),e[:,0].max(),e[0,0],e[:,1:].mean()],5))
 out=[]
 for k,q in coords2.items():
  l=base(q); p=quad(q)
  for n,f in [('quad',p),('quad+x',np.c_[p,quad(x)]),('trig',np.c_[p,np.sin(1.7*q),np.cos(1.7*q)]),('tanh',np.c_[p,np.tanh(1.7*q)]),('rat',np.c_[p,q/(1+q*q)])]:
   for a in [1e-5,.002,.03,.2,1.]:
    sc=score(f,a); out.append((*sc,k,n,a))
 print('best y',sorted(out)[:16]); print('best traj',sorted(out,key=lambda a:a[3])[:12]); print('best 52',sorted(out,key=lambda a:a[2])[:10])
