from graph_probe import *
from scipy.optimize import least_squares

def nonlinear(p,key):
 u=p['u']; v=p[key]; best=None
 for variant in ['shared','separate','shift']:
  k=1 if variant=='shared' else 4
  def design(q,r):
   w=q*r[:k]+(r[k:] if len(r)>k else 0)
   return np.c_[np.ones(len(q)),q,np.tanh(w) if key=='v1' else 1/(1+w*w)]
  def err(r):
   D=design(u,r); return (D@np.linalg.lstsq(D,v,rcond=None)[0]-v).ravel()
  r0=np.r_[np.full(k,2.2 if key=='v1' else .4),np.zeros(4 if variant=='shift' else 0)]
  opt=least_squares(err,r0,bounds=(np.r_[np.full(k,.05),np.full(len(r0)-k,-2)],np.r_[np.full(k,5),np.full(len(r0)-k,2)]),max_nfev=120)
  D=design(u,opt.x); c=np.linalg.lstsq(D,v,rcond=None)[0]; score=np.mean(abs(err(opt.x))); print(key,variant,score,opt.x)
  if best is None or score<best[0]*.65:best=(score,opt.x,c,design)
 score,r,c,d=best; return lambda q:d(q,r)@c

def makeframes(x):
 ff=frames(x); c1=nonlinear(s['panel_c'],'v1');c2=nonlinear(s['panel_c'],'v2'); out={'x':x}
 for nm in ('x','a-','a+','xb+','xb-','a-b+','a-b-','a+b+','a+b-'):
  q=ff[nm]; out[nm]=q
  for j,c in [(1,c1),(2,c2)]:
   w=c(q);out[nm+str(j)]=w;out[nm+str(j)+'b']=forward(w,B);out[nm+str(j)+'mix']=np.c_[q,w]
  out[nm+'12']=np.c_[c1(q),c2(q)]
 return out
if __name__=='__main__':
 from scipy.spatial.distance import cdist
 out=makeframes(x); results=[]
 for nm,q in out.items():
  w=(q-q.mean(0))/q.std(0); ds=cdist(w,w,'sqeuclidean')/w.shape[1]; D=np.c_[np.ones(len(w)),w]; D2=np.c_[D,np.column_stack([w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(w.shape[1]),2)])]
  for trend,M in [(1,D),(2,D2)]:
   for e in [.35,.6,1,1.7,3]:
    for r in [.0001,.008,.08]:
     K=np.exp(-ds/(2*e*e))+r*np.eye(len(w)); sc=scores(K,M);results.append((sc,nm,trend,e,r))
  K=D2@D2.T
  for r in [.00001,.003,.03,.3,3]:results.append((scores(K+r*np.eye(len(w)),D),nm,'poly',0,r))
 for j in range(2):
  print('TARGET',j)
  for sc,*cfg in sorted(results,key=lambda k:.35*k[0][0,j]+.35*k[0][1:,j].mean()+.30*k[0][1:,j].max())[:14]:print(cfg,np.round(sc[:,j],5))
