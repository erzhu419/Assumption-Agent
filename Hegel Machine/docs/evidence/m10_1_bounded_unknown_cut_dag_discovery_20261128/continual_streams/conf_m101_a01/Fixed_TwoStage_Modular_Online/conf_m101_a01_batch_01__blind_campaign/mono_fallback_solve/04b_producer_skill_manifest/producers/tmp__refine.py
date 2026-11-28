import sys
sys.path.insert(0,'/holdout_task')
from batch_solver import _aff,_profile
import pickle,itertools,numpy as np
from scipy.linalg import cho_factor,cho_solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z,_=_profile(tr['traj'],s['eval_times']); n=len(x); R=np.random.default_rng(318); splits=[(np.arange(k),np.arange(k,n)) for k in (52,56)]+[(p[:52],p[52:]) for p in [R.permutation(n) for _ in range(8)]]
def aff(p):return np.linalg.lstsq(_aff(p['u']),p['v'],rcond=None)[0]
pa=s['panel_a']; a=aff({k:v[np.linalg.norm(pa['v']-pa['u'],axis=1)<1] for k,v in pa.items()}); b=aff(s['panel_b']); A=(x-a[0])@np.linalg.inv(a[1:]); B=(x-b[0])@np.linalg.inv(b[1:]); C=_aff(A)@b
pairs=np.array(list(itertools.combinations_with_replacement(range(4),2))); q=(x[:,pairs[:,0]]*x[:,pairs[:,1]]); D=_aff(x); Q=np.c_[D,q];
print('quad',np.linalg.lstsq(Q,np.c_[y,z],rcond=None)[0].T.round(3))
out=[]
def one(K,trend,i,j,u):
 W=trend[i]; cf=cho_factor(K[np.ix_(i,i)]); F=cho_solve(cf,np.c_[W,u[i]]); w=F[:,:W.shape[1]]; a=F[:,-1]; p=np.linalg.solve(W.T@w,W.T@a); return trend[j]@p+K[np.ix_(j,i)]@(a-w@p)
def record(cfg,P):
 E=np.array([np.mean(abs(p-y[j])) for p,(_,j) in zip(P,splits)]); out.append((cfg,E,P))
links=[('id',y,lambda p:p),('asinh',np.arcsinh(y),np.sinh)]
for cap in [1.,1.4,2.3,4.]:
 links.extend([(f'sqrt{cap}',np.sqrt(cap-y),lambda p,c=cap:c-p*p),(f'log{cap}',np.log(cap-y),lambda p,c=cap:c-np.exp(p))])
for name,v in [('x',x),('A',A),('B',B),('C',C),('BC',np.c_[B,C]/2**.5)]:
 for warp,V in [('id',v),('tanh',np.tanh(.7*v)),('sin',np.sin(.9*v)),('asinh',np.arcsinh(v)),('augment',np.c_[v,np.tanh(1.6*v),.5*v*v])]:
  ds=(V[:,None,:]-V[None,:,:])**2
  for mode,q2 in [('full',ds.sum(2)),('add',ds),('pair',np.array([ds[:,:,i]+ds[:,:,j] for i,j in itertools.combinations(range(V.shape[1]),2)]).transpose(1,2,0))] if warp=='id' and name!='BC' else [('full',ds.sum(2))]:
   for kind in ('sq','m52','rq'):
    for ell in (.6,1.,1.6,2.5,4.):
     w=q2/ell**2; r=np.sqrt(w); K={'sq':np.exp(-w/2),'m52':(1+r+w/3)*np.exp(-r),'rq':1/(1+w/2)}[kind]; K=K if K.ndim==2 else K.mean(2)
     for lam in (.000003,.0003,.008):
      L=K+lam*np.eye(n)
      for dnm,H in [('aff',D),('quad',Q)]:
       if dnm=='quad' and warp!='id':continue
       for link,u,inverse in links:
        if (warp!='id' or mode!='full') and link not in ['id','asinh','sqrt1.4','log2.3']:continue
        P=[inverse(one(L,H,i,j,u)) for i,j in splits]; record((name,warp,mode,kind,ell,lam,dnm,link),P)
for rule in (lambda a:a[1].mean(),lambda a:a[1].max(),lambda a:np.mean(a[1][:2])):
 top=sorted(out,key=rule)[:12]; print('TOP');
 for cfg,E,P in top[:7]:print(cfg,E.round(4),round(E.mean(),4),round(E.max(),4))
 for k in (3,8,12):
  P=[np.mean([a[2][i] for a in top[:k]],axis=0) for i in range(len(splits))]; E=np.array([np.mean(abs(p-y[j])) for p,(_,j) in zip(P,splits)]); print('ens',k,E.round(4),E.mean(),E.max())
# Test an implicit rational response: y(1+x'Bx) = quadratic/cubic polynomial.
for vn,V in [('x',x),('A',A),('B',B),('C',C)]:
 sq=V*V; prod=np.column_stack([V[:,j]*V[:,k] for j,k in itertools.combinations_with_replacement(range(4),2)]); cube=np.column_stack([np.prod(x[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),3)]);
 for nm,F in [('squares',sq),('quadratic',prod),('full',np.c_[V,prod])]:
  for lam in (.01,.1,1,10):
   for deg in (2,3):
    H=Q if deg==2 else np.c_[Q,cube]; es=[]
    for i,j in splits:
     def err(w):
      h,k=w[:H.shape[1]],w[H.shape[1]:]; return np.r_[H[i]@h/(1+F[i]@k).clip(.2)-y[i],np.sqrt(lam)*k,.02*h]
     from scipy.optimize import least_squares
     w=least_squares(err,np.r_[np.linalg.lstsq(H[i],y[i],rcond=None)[0],np.zeros(F.shape[1])],max_nfev=30).x; p=H[j]@w[:H.shape[1]]/(1+F[j]@w[H.shape[1]:]).clip(.2);es.append(np.mean(abs(p-y[j])))
    if max(es[:2])<.055 or np.mean(es)<.042:print('rat',vn,nm,lam,deg,np.round(es,4))
