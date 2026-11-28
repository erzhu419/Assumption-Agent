from scalar_probe import *
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist

def mapping(x,q):
 sl,off=node_00._calibration(s['panel_a']); u=s['panel_b']['u']; W=np.linalg.lstsq(np.c_[np.ones(len(u)),u],s['panel_b']['v'],rcond=None)[0]; c=s['panel_c']; c1=node_00._operator(c['u'],c['v1'],'tanh',(.3,3)); c2=node_00._operator(c['u'],c['v2'],'rational',(.015,.8)); out={}
 for k,(v,z) in enumerate([(x,q),((x-off)/sl,(q-off)/sl),(x*sl+off,q*sl+off)]):
  for j,(f,g) in enumerate([(v,z),(v@W[1:]+W[0],z@W[1:]+W[0])]):
   out[f'{k}{j}']=(f,g)
   out[f'{k}{j}t']=(c1(f),c1(g)); out[f'{k}{j}r']=(c2(f),c2(g))
 return out

def nonlinear(x,q,F,G,y,kind,units):
 mu=x.mean(0); sd=x.std(0); x=(x-mu)/sd; q=(q-mu)/sd; d=x.shape[1]; Q=np.linalg.qr(F,mode='reduced')[0]; ry=y-Q@(Q.T@y)
 def fun(x,w):
  v=x@w.reshape(d+1,units)[:-1]+w.reshape(d+1,units)[-1];
  if kind=='tanh':return np.tanh(v)
  if kind=='sin':return np.c_[np.sin(v),np.cos(v)]
  return 1/(1+v*v)
 def res(w):
  h=fun(x,w); h-=Q@(Q.T@h); return ry-h@np.linalg.lstsq(h,ry,rcond=1e-9)[0]
 rng=np.random.default_rng(2026); best=None
 for i in range(14):
  init=rng.normal(0,.5 if i<9 else 1.2,(d+1)*units); o=least_squares(res,init,max_nfev=140,bounds=(-6,6)); key=np.sqrt(np.mean(o.fun**2));
  if best is None or key<best[0]:best=(key,o.x)
 key,w=best; beta=np.linalg.lstsq(np.c_[F,fun(x,w)],y,rcond=1e-9)[0]; return np.c_[G,fun(q,w)]@beta,key,w,beta

for tr,va in [(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20)),(np.arange(72)[np.arange(72)%3!=1],np.arange(72)[np.arange(72)%3==1])]:
 p,state,y,v,t,x,q=fold(tr,va); a,b=p['scores'],state@p['modes'].T; out=mapping(x,q); print('FOLD',va[:3],flush=True); records=[]
 def record(k,f,g):
  f=np.c_[np.ones(len(x)),f];g=np.c_[np.ones(len(q)),g]; U,L,V=np.linalg.svd(f,full_matrices=False)
  for rid in [0,.001,.01,.1]:
   w=V.T@(((L/(L**2+rid)))*(U.T@y)); e=y-f@w; loo=np.mean(np.abs(e/np.maximum(1-(U**2)@(L**2/(L**2+rid)),1e-8))); cv=np.mean(abs(v-g@w)); records.append((cv,loo,k,rid,np.mean(abs(e))))
 for k,(f,g) in out.items():
  F=np.c_[f,expand(f,2)];G=np.c_[g,expand(g,2)]; record(k+'quad+state',np.c_[F,a],np.c_[G,b]); record(k+'quad+statequad',np.c_[F,a,expand(a,2), (f[:,:,None]*a[:,None,:]).reshape(len(f),-1)],np.c_[G,b,expand(b,2),(g[:,:,None]*b[:,None,:]).reshape(len(g),-1)]); record(k+'rawquad+coord+state',np.c_[x,expand(x,2),f,a],np.c_[q,expand(q,2),g,b]);
  for j in [.3,.5,.8,1.,1.5,2.]:
   for tag,func in [('t',np.tanh),('s',np.sin),('r',lambda u:1/(1+u*u))]:record(f'{k}{tag}{j}',np.c_[x,expand(x,2),func(j*f),a],np.c_[q,expand(q,2),func(j*g),b]);
 print('best',sorted(records)[:8]); print('loo',sorted(records,key=lambda z:z[1])[:5],flush=True)
 for k,(f,g) in [('x',(x,q)),('x+a',(np.c_[x,a],np.c_[q,b])),('a',(a,b))]:
  for tag in ['tanh','sin','rat']:
   for units in ([1,2] if k=='x' else [1]):
    F=np.c_[np.ones(len(x)),x,expand(x,2),a]; G=np.c_[np.ones(len(q)),q,expand(q,2),b]; pred,err,w,coef=nonlinear(f,g,F,G,y,tag,units); print(k,tag,units,round(np.mean(abs(v-pred)),6),round(err,6),np.round(w,2),flush=True)
