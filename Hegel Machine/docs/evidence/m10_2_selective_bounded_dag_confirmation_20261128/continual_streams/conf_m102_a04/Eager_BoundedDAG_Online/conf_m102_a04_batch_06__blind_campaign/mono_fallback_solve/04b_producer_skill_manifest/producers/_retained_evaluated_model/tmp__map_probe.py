import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']; y=tr['y']; Y=np.c_[y,T]; n=len(y)
np.set_printoptions(precision=4,suppress=True,linewidth=155)
def lin(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def one(a):return np.c_[np.ones(len(a)),a]
def mon(a,idx):return np.column_stack([np.prod(a[:,j],axis=1) for j in idx])
pairs=list(itertools.combinations(range(4),2))
cyc=[(0,1),(1,2),(2,3),(3,0)]
def ca(a):return np.c_[one(a),mon(a,cyc),a**3]
A=s['panel_a']; u,v=A['u'],A['v']; keep=abs(v).max(1)<4; c=lin(ca(u[keep]),v[keep]); c[abs(c)<.001]=0
print('t/telemetry',t,s['telemetry_vec'],'calib',c)
def inv(x):
 return np.array([least_squares(lambda a:(ca(a[None])@c).ravel()-q,np.linalg.solve(c[1:5].T,q-c[0]),gtol=1e-11).x for q in x])
z=inv(tr['x']); b=s['panel_b']; cb=lin(one(b['u']),b['v']); B=lambda x:one(x)@cb
p=s['panel_c']; C1=lambda x:np.c_[np.ones(len(x)),mon(x,cyc)]; c1=lin(C1(p['u']),p['v1']); C=lambda x:C1(x)@c1

def c2fit(u,v):
 for name,fn in [('expquad',lambda a,k:np.exp(-k*a*a)),('cos',lambda a,k:np.cos(k*a)),('rational',lambda a,k:1/(1+k*a*a)),('tanh',lambda a,k:np.tanh(k*a))]:
  def f(a,k):return np.c_[one(a),fn(a,k)]
  opt=least_squares(lambda k:(f(u,k)@lin(f(u,k),v)-v).ravel(),[1],bounds=(.02,6)); k=opt.x; F=f(u,k); co=lin(F,v); er=np.sqrt(np.mean((F@co-v)**2)); print('C2 compact',name,k,er); yield er,lambda a,f=f,k=k,co=co:f(a,k)@co
C2=min(c2fit(p['u'],p['v2']),key=lambda a:a[0])[1]
folds=[(np.arange(52),np.arange(52,72))]+[(i[:52],i[52:]) for i in [np.random.default_rng(j).permutation(n) for j in range(4)]]
def cv(K,l):
 E=[]
 for a,b in folds:
  d=abs(K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+l*np.eye(len(a)),Y[a],assume_a='pos')-Y[b]);E.append([d[:,0].mean(),d[:,1:].mean()])
 return np.array(E)
def kern(X):
 S=np.maximum(.1,X.std(0)); X=(X-X.mean(0))/S; G=X@X.T/X.shape[1];D=cdist(X,X,'sqeuclidean')/X.shape[1];return G,D
zz=B(z); dd={'z':z,'b':zz,'c':C(z),'d':C2(z),'cb':C(zz),'db':C2(zz),'xyz':tr['x'],'z+c+d':np.c_[z,C(z),C2(z)],'b+cb+db':np.c_[zz,C(zz),C2(zz)],'c+d':np.c_[C(z),C2(z)],'cb+db':np.c_[C(zz),C2(zz)]}
for name,X in dd.items():
 out=[]
 for i,F in enumerate([X,np.tanh(X),np.sin(X),X/(1+abs(X)),np.c_[X,np.sin(2*X),np.cos(2*X)],np.c_[X,np.tanh(2*X)]]):
  G,D=kern(F)
  for p in (1,2,3,4):
   for amp,k in [(0,1),(.1,.25),(.1,1),(1,.25),(1,1)]:
    K=(1+.45*G)**p+amp*np.exp(-k*D)
    for l in [1e-6,1e-4,.002,.02]:
     e=cv(K,l);out.append((e.mean(0),e[0],e.max(0),(i,p,amp,k,l)))
 print(name,'best-y',min(out,key=lambda a:a[0][0]),'best-T',min(out,key=lambda a:a[0][1]));
print('z and outputs sample',np.c_[z,y,T[:,0],T[:,-1]][:15]); print('correlations by source');
for name,X in dd.items(): print(name,np.corrcoef(np.c_[X,y,T[:,0],T[:,-1]].T)[:-3,-3:].T)
