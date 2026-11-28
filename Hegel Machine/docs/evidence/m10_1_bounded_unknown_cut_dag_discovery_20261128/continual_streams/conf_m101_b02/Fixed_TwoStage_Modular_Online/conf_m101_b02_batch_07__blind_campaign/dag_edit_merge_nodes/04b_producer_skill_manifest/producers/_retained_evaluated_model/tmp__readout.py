import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; A,B,C=[s['panel_'+k] for k in 'abc']; X=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; n=len(X)
def ls(d,y):return np.linalg.lstsq(d,y,rcond=1e-9)[0]
def aff(x):return np.column_stack((np.ones(len(x)),x))
def pairs(z):return np.column_stack([z[:,i]*z[:,j] for i,j in ((0,1),(0,3),(1,2),(2,3))])
def panel(p,key,d,start):
 def r(k):
  h=d(p['u'],k);return (h@ls(h,p[key])-p[key]).ravel()
 k=least_squares(r,start,max_nfev=150).x if len(start) else []
 w=ls(d(p['u'],k),p[key]);return np.std(r(k)),lambda x:d(x,k)@w,k
D=aff(A['u']); c=np.vstack((np.zeros(4),np.eye(4)))
for _ in range(10):
 w=1/np.sqrt(1+np.sum((D@c-A['v'])**2,axis=1)/.02**2);c=ls(D*w[:,None],A['v']*w[:,None])
_,h,_=panel(C,'v2',lambda z,p:aff(np.c_[z,np.sqrt(1+p[0]*z*z)]),[.9]);_,cc,_=panel(C,'v1',lambda z,p:aff(pairs(z)),[])
bs=[]
for nm,fun,start in [('sin',lambda z,k:z+k[1]*np.sin(k[0]*z),[2.8,-.4]),('exp',lambda z,k:z+k[1]*z*np.exp(-k[0]*z*z),[1,-1]),('rat',lambda z,k:z+k[1]*z/(1+k[0]*z*z),[.75,-1]),('tanh',lambda z,k:z+k[1]*np.tanh(k[0]*z),[1.4,-.76])]:
 for even in ['none','cross','square','evenroot']:
  def d(z,p,fun=fun,even=even):
   o=fun(z,p);e={'none':lambda:np.empty((len(z),0)),'cross':lambda:pairs(z),'square':lambda:z*z,'evenroot':lambda:np.sqrt(1+p[2]*z*z)}[even]();return aff(np.c_[o,e])
  er,b,k=panel(B,'v',d,start+([.9] if even=='evenroot' else []));bs.append((er,nm,even,b));print('B',nm,even,er,np.round(k,4))
bs.sort(key=lambda z:z[0]);b=bs[0][-1]; print('B chosen',bs[0][:3]);
for nm,fn,start in [('exp2',lambda k:np.column_stack([-np.expm1(-v*t) for v in k]),[.4,.7]),('exp+t',lambda k:np.c_[-np.expm1(-k[0]*t),t],[.45]),('exp+sin',lambda k:np.c_[-np.expm1(-k[0]*t),np.sin(k[1]*t)],[.45,1]),('exp+cos',lambda k:np.c_[-np.expm1(-k[0]*t),1-np.cos(k[1]*t)],[.45,1]),('exp+offset',lambda k:np.c_[-np.expm1(-k[0]*t),np.ones(len(t))],[.5])]:
 def r(k):
  d=fn(k);return (T.T-d@ls(d,T.T)).ravel()
 p=least_squares(r,start,bounds=(.001,5)).x;print('time',nm,p,np.std(r(p)))
# Use orthogonal time modes to compare scalar and dominant trajectory readouts on equivalent scales.
_,_,V=np.linalg.svd(T,full_matrices=False); Y=np.c_[y,T@V[:2].T]; scale=np.std(Y,axis=0); Y/=scale
folds=[np.arange(52,72),*np.array_split(np.arange(72),4)]
def metric(p,v):
 R=abs(p*scale-np.c_[y,T@V[:2].T][v]);return R[:,0].mean(),np.mean(abs((p[:,1:]*scale[1:])@V[:2]-T[v]))
def basis(z,bb):
 g=bb(z); j=cc(z); k=h(z); u=np.c_[z,g,j,k]
 return {'z':z,'b':g,'c':j,'h':k,'A':np.c_[z,g,j,k],'Q':np.c_[z,g,j,k,pairs(z),z*z], 'q':np.c_[z,pairs(z),z*z], 'zb':np.c_[z,g], 'zh':np.c_[z,k], 'bc':np.c_[g,j], 'bh':np.c_[g,k]}
def main():
 from sklearn.linear_model import OrthogonalMatchingPursuit,ElasticNet
 from sklearn.preprocessing import StandardScaler
 results=[]; rd=[]
 for ci,z in enumerate([X,np.linalg.solve(c[1:].T,(X-c[0]).T).T,X@c[1:]+c[0]]):
  G=basis(z,b);zu=(z-z.mean(0))/z.std(0)
  blocks=[G['A'],*[np.column_stack([np.prod(z[:,j],axis=1) for j in it.combinations_with_replacement(range(4),k)]) for k in range(2,6)]]
  for oo in [G['b'],G['c'],G['h']]:blocks.append((z[:,:,None]*oo[:,None,:]).reshape(n,-1))
  for g in [z,*[G[q] for q in 'bch'],np.column_stack([z[:,i]+a*z[:,j] for i,j in it.combinations(range(4),2) for a in (-1,1)])]:
   for f in [.6,1,1.7,2.8]:blocks.extend([np.sin(f*g),np.cos(f*g),np.tanh(f*g)])
  for ki,D in enumerate([np.column_stack(blocks),np.column_stack(blocks[:8]),np.column_stack(blocks[:4]),np.column_stack([G['A'],*blocks[8:]])]):
   for algo,param in [(OrthogonalMatchingPursuit,j) for j in [5,8,11,15,20]]+[(ElasticNet,j) for j in [.00007,.0003,.001,.003,.009]]:
    er=[]
    for v in folds:
     i=np.setdiff1d(np.arange(n),v);ss=StandardScaler().fit(D[i]);a=ss.transform(D[i]);q=ss.transform(D[v]); P=[]
     for j in range(3):
      m=algo(n_nonzero_coefs=param) if algo is OrthogonalMatchingPursuit else algo(alpha=param,l1_ratio=.88,max_iter=4000,tol=.0003)
      P.append(m.fit(a,Y[i,j]).predict(q))
     er.append(metric(np.column_stack(P),v))
    rd.append((np.max(er),np.max(er[0]),ci,ki,algo.__name__,param,np.round(er,4).tolist()))
  # Mixed additive and ridge projection nonlinearities, with shared projections estimated across outputs.
  for base,proj in [('z','z'),('Q','z'),('z','zb'),('q','zb'),('A','z'),('A','zb'),('zb','zb'),('Q','zh'),('A','bc'),('A','A')]:
   for kind,m in it.product(['tanh','sin','rat'],[1,2,3]):
    er=[]
    for v in folds:
     i=np.setdiff1d(np.arange(n),v); L=aff(G[base]); U=aff(G[proj]/np.maximum(G[proj][i].std(0),.15)); u=U[i]; l=L[i]; yy=Y[i]; rg=.003 if L.shape[1]>10 else .0003
     reg=np.sqrt(rg)*np.eye(L.shape[1]+m);reg[0,0]=0
     def make(u,p):
      w=u@p.reshape(-1,m);return np.tanh(w) if kind=='tanh' else np.sin(w) if kind=='sin' else w/(1+w*w)
     def weights(d):return ls(np.r_[d,reg],np.r_[yy,np.zeros((len(reg),3))])
     def resid(p):
      d=np.c_[l,make(u,p)]; w=weights(d);return np.r_[(d@w-yy).ravel(),(reg@w).ravel(), .022*p]
     rng=np.random.default_rng(501); p0s=[rng.normal(0,f/np.sqrt(u.shape[1]),m*u.shape[1]) for f in (.7,1.5,2.5)]
     best=min((least_squares(resid,p,max_nfev=50) for p in p0s),key=lambda a:a.cost);w=weights(np.c_[l,make(u,best.x)]);pr=np.c_[L[v],make(U[v],best.x)]@w;er.append(metric(pr,v))
    results.append((np.max(er),np.max(er[0]),ci,base,proj,kind,m,np.round(er,4).tolist()))
 print('SPARSE',*sorted(rd)[:8],sep='\n'); print('RIDGE PROJ',*sorted(results)[:12],sep='\n');print('PROJ SCALAR',*sorted(results,key=lambda r:max(e[0] for e in r[-1]))[:5],sep='\n');print('PROJ TRAJ',*sorted(results,key=lambda r:max(e[1] for e in r[-1]))[:5],sep='\n')
if __name__=='__main__':main()
