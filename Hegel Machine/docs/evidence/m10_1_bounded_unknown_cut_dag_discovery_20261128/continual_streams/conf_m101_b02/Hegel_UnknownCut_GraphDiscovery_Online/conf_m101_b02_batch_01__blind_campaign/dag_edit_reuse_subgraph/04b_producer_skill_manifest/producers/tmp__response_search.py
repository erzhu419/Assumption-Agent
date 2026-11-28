import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; print('study keys',list(s),flush=True)
D=lambda x:np.c_[np.ones(len(x)),x]
L=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
a=s['panel_a']; X=D(a['u']); p=L(X,a['v']);
for _ in range(12):
 w=1/np.sqrt(1+(np.linalg.norm(X@p-a['v'],axis=1)/.04)**2); p=L(X*w[:,None],a['v']*w[:,None]);
z=np.linalg.solve(p[1:].T,(s['train']['x']-p[0]).T).T
warps=[lambda u:u-11.0993*u*(-np.expm1(-(1.0088*u)**2)),lambda u:u/(1+.8265*abs(u)),lambda u:u+.9074*np.log(np.cosh(1.101*u))]
M=[]
for k,f in enumerate(warps):
 panel=s['panel_b' if k==0 else 'panel_c']; u=panel['u']; v=panel[['v','v1','v2'][k]]; g=(lambda x,f=f:np.c_[x,f(x)]) if k==1 else f
 A=L(D(g(u)),v); print('panel',k,'A',np.round(A,4),'resid rms',np.mean((D(g(u))@A-v)**2)**.5,flush=True); M.append(lambda x,g=g,A=A:D(g(x))@A)
 if k==0:
  F=np.c_[D(g(u)),u**2,u**3,*[u[:,i]*u[:,j] for i in range(4) for j in range(i)]]; print('B residual vs pair grammar',np.sqrt(np.mean((F@L(F,v)-v)**2)),flush=True)
t=s['eval_times']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y); folds=[np.arange(n)[-20:],*np.array_split(np.random.default_rng(81).permutation(n),4)]
U={'z':z};
for i,f in enumerate(M): U[str(i)]=f(z)
for i,f in enumerate(M):
 for j,g in enumerate(M): U[str(i)+str(j)]=g(f(z))
for i in (1,2):
 for j in (1,2): U['0'+str(i)+str(j)]=M[j](U['0'+str(i)])
sets={'z':['z'],'joint':['z','0','1','2'],'route':['z','0','1','2','01','02'],'route2':['z','0','01','02'],'all':list(U),'C':['1','2'],'BC':['0','01','02']}
for name,ks in sets.items():
 for basis in ('linear','poly','rbf','matern','mixed'):
  allerr=[]
  for h in ([.35,.7,1.4,2.8,5.6] if basis in ('rbf','matern','mixed') else [1.]):
   for lam in (.000001,.0001,.003,.03,.2):
    errs=[]
    for va in folds:
     tr=np.setdiff1d(np.arange(n),va); us=[(U[k]-U[k][tr].mean(0))/(U[k][tr].std(0)+1e-9) for k in ks]; x=np.concatenate(us,axis=1); dot=x@x.T/x.shape[1]; d=cdist(x,x,'sqeuclidean')/x.shape[1]; K=1+dot
     if basis=='poly': K=K**3
     if basis=='rbf': K+=np.exp(-d/(2*h*h))
     if basis=='matern': r=np.sqrt(5*d)/h; K+=(1+r+r*r/3)*np.exp(-r)
     if basis=='mixed':
      for a in us:
       r=cdist(a,a,'sqeuclidean')/a.shape[1]; K+=np.exp(-r/(2*h*h))/len(us)
      K+=np.exp(-d/(2*h*h))
     e=abs(K[np.ix_(va,tr)]@solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),Y[tr],assume_a='pos')-Y[va]); errs.append([e[:,0].mean(),e[:,1:].mean()]);
    allerr.append((np.max(errs),h,lam,np.array(errs)[0],np.max(errs,axis=0)))
  print(name,basis,min(allerr,key=lambda r:r[0]),flush=True)
# Check whether calibrated C1/C2 routes describe a latent switch.
for subset in (['01','02'],['1','2']):
 for g in [z,*U.values()]:
  def calc(q):
   w=1/(1+np.exp(-np.clip(D(g)@q,-40,40))); a,b=[U[k] for k in subset]; return D(np.c_[z,U['0'],a*w[:,None],b*(1-w[:,None])])
  errs=[]
  for va in folds:
   tr=np.setdiff1d(np.arange(n),va)
   def fun(q):
    X=calc(q); return np.r_[(X[tr]@L(X[tr],Y[tr])-Y[tr]).ravel(),.025*q]
   fits=[least_squares(fun,r,max_nfev=70) for r in (np.zeros(5),np.array([0.,1.,-1.,.5,.5]),np.array([0.,-1.,1.,-.5,-.5]))]; X=calc(min(fits,key=lambda r:r.cost).x); e=abs(X[va]@L(X[tr],Y[tr])-Y[va]); errs.append([e[:,0].mean(),e[:,1:].mean()])
  print('switch',subset,np.round(errs[0],5),np.round(np.max(errs,axis=0),5))
