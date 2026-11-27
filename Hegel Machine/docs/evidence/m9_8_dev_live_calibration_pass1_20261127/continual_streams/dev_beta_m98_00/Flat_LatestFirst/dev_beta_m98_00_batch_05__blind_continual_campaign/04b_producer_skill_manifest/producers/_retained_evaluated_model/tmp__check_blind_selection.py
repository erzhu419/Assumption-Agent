exec(open('/tmp/wall_diag.py').read().split("F={'base'")[0])
from scipy.optimize import minimize_scalar
from functools import lru_cache
ij=np.asarray(s['sensor_ij']); count=len(ij); loc=np.arange(len(y))%count; shift=np.arange(len(y))//count
patterns={'nominal': [], 'x-pair': [0,1], 'x-high': [1], 'common': [0,1,4,5]}
@lru_cache(maxsize=4000)
def design(name,beta):
 wall=np.ones(8); wall[patterns[name]]=np.exp(beta)
 return np.c_[m._response(s,wall,src)[0],z]
def eval_fit(name,mask,target=y):
 def obj(beta,full=False):
  a=design(name,beta); am=a[mask]
  q=lsq_linear(np.vstack((am,10*np.r_[np.ones(n),np.zeros(nz)])),np.r_[target[mask],10*total],bounds=bounds,method='bvls',tol=1e-10).x
  d=a@q-target
  return (beta,d,q) if full else d[mask]@d[mask]
 beta=0. if name=='nominal' else minimize_scalar(obj,bounds=(-.6,.6),method='bounded',options={'xatol':1e-5}).x
 return obj(beta,True)
for name in patterns:
 b,e,q=eval_fit(name,allrows); print(name,'beta',round(b,5),'SSE',round(e@e,8))
for title,outer in [('rosette',[folds==i for i in range(4)]),('shift',[shift==i for i in range(3)]),('wall', [ij[loc,0]==0, ij[loc,1]==0, ij[loc,0]>=10, ij[loc,1]>=8])]:
 scores=[]
 for h in outer:
  inner=[(folds==i)&~h for i in range(4)]; ranking=[]
  for name in patterns:
   errs=[eval_fit(name,~(h|v))[1][v] for v in inner if v.any()]
   ranking.append(np.mean(np.concatenate(errs)**2))
  name=list(patterns)[np.argmin(ranking)]; p=eval_fit(name,~h)[1][h]; base=eval_fit('nominal',~h)[1][h]
  scores.append((name,round(np.mean(p*p)**.5,6),round(np.mean(base*base)**.5,6)))
 print(title,scores)
_,r,q=eval_fit('nominal',allrows)
base=design('nominal',0.)@q; actual=r@r-min(eval_fit(name,allrows)[1]@eval_fit(name,allrows)[1] for name in patterns)
rng=np.random.default_rng(812); null=[]
for k in range(149):
 sign=rng.choice([-1.,1.],size=count//4)[loc//4]; yy=base+r*sign
 e=eval_fit('nominal',allrows,yy)[1]; a=e@e
 null.append(a-min((lambda t:t@t)(eval_fit(name,allrows,yy)[1]) for name in list(patterns)[1:]))
print('cluster sign bootstrap max improvement',round(actual,8),'empirical exceedance',(1+sum(t>=actual for t in null))/(1+len(null)))
