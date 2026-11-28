import pickle,numpy as np,itertools
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def af(x):return np.c_[np.ones(len(x)),x]
def fit(a,y):return np.linalg.lstsq(a,y,rcond=None)[0]
def proj(a,y):return y-a@fit(a,y)
def loo(a,y): return np.sqrt(np.mean((proj(a,y)/(1-(np.linalg.qr(a)[0]**2).sum(1))[:,None])**2))
def cal(x):return np.c_[af(x),x*np.roll(x,-1,axis=1)]
a=S['panel_a'];u,v=a['u'],a['v'];w=fit(af(u),v)
for _ in range(25):
 q=np.minimum(1,.05/np.maximum(np.linalg.norm(v-af(u)@w,axis=1),1e-12))**.5;w=fit(af(u)*q[:,None],v*q[:,None])
k=np.linalg.norm(v-af(u)@w,axis=1)<2;u,v=u[k],v[k]
from scipy.optimize import minimize_scalar
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from scipy.linalg import solve
T=S['train']; x=T['x']; t=S['eval_times']; traj=T['traj'];
def am(q):return np.c_[af(q),q*np.roll(q,-1,axis=1),q**3]
A=fit(am(u),v); xi=x.copy()
for i in range(4):
 f=lambda q: np.c_[np.ones(len(q)),q[:,i],q[:,(i+1)%4],q[:,i]*q[:,(i+1)%4],q[:,i]**3]; print('A exact',i,loo(f(u),v[:,i:i+1]))
xi=least_squares(lambda z:(am(z.reshape(-1,4))@A-x).ravel(),xi.ravel(),max_nfev=20).x.reshape(-1,4)
print('inverse ranges',xi.min(0),xi.max(0))
B=fit(af(S['panel_b']['u']),S['panel_b']['v']); print('B gram',B[1:].T@B[1:]);
C=S['panel_c']; cu=C['u']; ck=[]
for j,fn in enumerate([np.tanh,np.cos],1):
 cv=C['v'+str(j)]; o=minimize_scalar(lambda f: np.mean(proj(np.c_[af(cu),fn(f*cu)],cv)**2),bounds=(.6,4),method='bounded'); ck.append((fn,o.x,fit(np.c_[af(cu),fn(o.x*cu)],cv)))
def Cj(q,j):
 fn,f,w=ck[j];return np.c_[af(q),fn(f*q)]@w
X={'raw':x,'U':xi};
for name,z in list(X.items()):
 X[name+'B']=af(z)@B; X[name+'Bi']=(z-B[0])@np.linalg.inv(B[1:]); X[name+'C']=Cj(z,0); X[name+'D']=Cj(z,1)
for name in ['UB','UBi']:
 X[name+'C']=Cj(X[name],0);X[name+'D']=Cj(X[name],1)
for rates in [[.412,1.814]]:
 H=-np.expm1(-t[:,None]*rates); modes=fit(H,traj.T).T
Y=np.c_[T['y'],modes]; print('Y mean,std,corr',Y.mean(0),Y.std(0),np.corrcoef(Y.T)); print('linear y~modes',fit(af(modes),Y[:,0]),np.mean(np.abs(proj(af(modes),Y[:,0])))); print('summary',T['summary'])
sp=[(np.arange(52),np.arange(52,72)),*list(KFold(4,shuffle=True,random_state=123).split(x))]
def score(F,reg):
 out=[]
 for a,b in sp:
  sc=StandardScaler().fit(F[a]); q=sc.transform(F); y=Y[a]; ym=y.mean(0); ys=y.std(0); q0=q[a]; z=(y-ym)/ys
  if isinstance(reg,float):
   zhat=q[b]@(q0.T@solve(q0@q0.T+reg*np.eye(len(a)),z,assume_a='pos'))
  else:
   zhat=np.column_stack([ElasticNet(alpha=reg[0],l1_ratio=reg[1],max_iter=3500,tol=2e-4).fit(q0,zz).predict(q[b]) for zz in z.T]);
  ep=ym+zhat*ys-Y[b]; out.append(np.r_[np.mean(abs(ep),0),np.mean(abs(ep[:,1:]@H.T))]);
 return np.array(out)
def phi(q,k):
 ps=[q[:,i]*q[:,j] for i,j in itertools.combinations(range(4),2)];
 if k==0:return q
 if k==1:return np.c_[q,q*q,np.array(ps).T]
 if k==2:return np.c_[phi(q,1),*[np.sin(f*q) for f in [1,2,3]],*[np.cos(f*q) for f in [1,2,3]]]
 p=np.array([q[:,i]+sg*q[:,j] for i,j in itertools.combinations(range(4),2) for sg in [1,-1]]).T
 return np.c_[phi(q,2),np.array([np.prod(q[:,ids],axis=1) for ids in itertools.combinations(range(4),3)]).T,*[np.sin(f*p) for f in [1,2,3]],*[np.cos(f*p) for f in [1,2,3]],np.tanh(2*q)]
print('columns: CV52 y, am1, am2, traj; randomized CV means (same)')
for nm,q in X.items():
 for k in range(4):
  F=phi(q,k); regs=[.02,.3,2.,(.003,.8),(.015,.8)]; es=[score(F,r) for r in regs]; e,r=min(zip(es,regs),key=lambda er:np.mean(er[0][1:][:,[0,3]])); print(nm,k,r,np.round(e[0],4),np.round(e[1:].mean(0),4))
for nms in [('U','UB'),('U','UC','UD'),('U','UB','UC','UD'),('U','UB','UBC','UBD'),tuple(X)]:
 for k in [1,2,3]:
  F=np.concatenate([phi(X[nm],k) for nm in nms],1); regs=[.02,.3,2.,(.003,.8),(.015,.8)]; es=[score(F,r) for r in regs]; e,r=min(zip(es,regs),key=lambda er:np.mean(er[0][1:][:,[0,3]])); print(nms,k,r,np.round(e[0],4),np.round(e[1:].mean(0),4))
