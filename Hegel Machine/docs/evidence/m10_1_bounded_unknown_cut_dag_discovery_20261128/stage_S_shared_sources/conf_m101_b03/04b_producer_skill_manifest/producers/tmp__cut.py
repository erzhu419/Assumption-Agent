import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe import *
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.optimize import least_squares
from sklearn.linear_model import OrthogonalMatchingPursuit
from collections import defaultdict
print('k',k,'reconstruction',np.mean(abs(z@f.T-T),axis=0)); U,S,W=np.linalg.svd(T,full_matrices=False); print('trajectory modes',S,W[:3]); print('mean traj',T.mean(0));
def mp(r,kind):
 if kind==0: return np.c_[1-np.exp(-r[0]*t),1-np.exp(-r[1]*t)]
 if kind==1: return np.c_[np.exp(-r[0]*t),np.exp(-r[1]*t)]
 if kind==2: return np.c_[1-np.exp(-r[0]*t), t*np.exp(-r[1]*t)]
 if kind==3: return np.c_[np.exp(-r[0]*t), 1-np.exp(-r[1]*t)]
for kind in range(4):
 def fun(r):
  h=mp(r,kind); return (h@ls(h,T[:n].T)-T[:n].T).ravel()
 r=least_squares(fun,[.5,2],bounds=(.0001,10)).x; h=mp(r,kind); print('time law',kind,r,np.mean(abs(fun(r))),np.mean(abs(h@ls(h,T[n:].T)-T[n:].T)))
Z=np.c_[y,U[:,:2]*S[:2]]
Y=Z/Z[:n].std(0); results=defaultdict(list)
for name in ('ai','aib','aibi','x'):
 q=base[name]; D=cdist(q,q,'sqeuclidean'); 
 for deg in [2,3,4,5]:
  for scale in [.3,1,3]:
   K=(1+scale*q@q.T)**deg
   for a in [1e-6,1e-4,.01,.1]:
    P=K[n:,:n]@solve(K[:n,:n]+a*np.eye(n),Y[:n],assume_a='pos'); results['poly',name].append((np.mean(abs(P-Y[n:]),axis=0), (deg,scale,a)))
 for g in [.005,.015,.04,.10,.25,.5,1,2]:
  for additive in [False,True]:
   for a in [1e-8,1e-6,1e-4,.001,.01]:
    K=np.exp(-g*D) if not additive else sum(np.exp(-g*(q[:,i,None]-q[None,:,i])**2) for i in range(4)); K+=1
    P=K[n:,:n]@solve(K[:n,:n]+a*np.eye(n),Y[:n],assume_a='pos');results['rbf',name].append((np.mean(abs(P-Y[n:]),axis=0),(g,additive,a)))
for key,v in results.items(): print('kernel',key,[(round(min(v,key=lambda r:r[0][j])[0][j]*Z[:n,j].std(),5),min(v,key=lambda r:r[0][j])[1]) for j in range(3)])
def dictionary(q,tag):
 p=PolynomialFeatures(3); F=p.fit_transform(q)[:,1:]; names=list(p.get_feature_names_out())[1:]; fs=[*F.T];
 dirs=[(str(i),q[:,i]) for i in range(4)]
 for i,j in itertools.combinations(range(4),2):
  for c in [-2,-1,-.5,.5,1,2]: dirs.append((f'{i}+{c}*{j}',q[:,i]+c*q[:,j]))
 for a,d in dirs:
  for f in [np.abs,np.tanh,lambda x: np.sin(2*x),lambda x: np.cos(2*x),lambda x: x/(1+x*x),lambda x: 1/(1+x*x)]:
   fs.append(f(d)); names.append(f'{a} {f.__name__}')
  for h in [-.3,0,.3]:
   fs.append(np.maximum(d-h,0)); names.append(f'{a} hinge {h}')
 F=np.array(fs).T; m=F[:n].mean(0); sd=F[:n].std(0); F=(F-m)/np.maximum(sd,.000001); G=Y-Y[:n].mean(0);
 for j in range(Y.shape[1]):
  scores=[]
  for num in [4,8,12,16,20,25]:
   model=OrthogonalMatchingPursuit(n_nonzero_coefs=num,fit_intercept=False).fit(F[:n],G[:n,j]); sel=np.flatnonzero(model.coef_); scores.append((np.mean(abs(model.predict(F[n:])-G[n:,j]))*Z[:n,j].std(),num,[(names[i],round(model.coef_[i],2)) for i in sel if abs(model.coef_[i])>.15]))
  print('sparse',tag,j,min(scores))
for name in ('ai','aib','aibi','x'): dictionary(base[name],name)
p=s['panel_c']; dictionary(p['u'],'noop') if False else None
