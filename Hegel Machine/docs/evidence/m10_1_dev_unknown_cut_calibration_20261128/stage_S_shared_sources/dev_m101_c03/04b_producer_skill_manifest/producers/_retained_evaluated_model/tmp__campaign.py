import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=4,suppress=True,linewidth=150)
def ls(X,Y):return np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def pair(x): return np.column_stack([x[:,i]*x[:,j] for i,j in it.combinations(range(4),2)])
def rms(e):return np.sqrt(np.mean(e*e))
def calfit(a):
 u,v=a['u'],a['v'];i,j=np.triu_indices(len(u),1);m=np.median((v[i]-v[j])/(u[i]-u[j]),axis=0);d=np.median(v-m*u,axis=0);return np.array([ls(aff(u[ok,k,None]),v[ok,k]) for k in range(4) for ok in [abs(v[:,k]-d[k]-m[k]*u[:,k])<.02]])
u=s['panel_c']['u'];v=s['panel_c']['v1'];v2=s['panel_c']['v2']; jj=np.arange(len(u));
def cfit(u,v,kind):
 def f(z,p):
  if kind=='tanh':return aff(np.c_[z,np.tanh(z*p[:4]+p[4:8])])
  if kind=='tied':return aff(z*p[8]+np.tanh(z*p[:4]+p[4:8]))
  if kind=='sin':return aff(np.c_[z,np.sin(z*p[:4]+p[4:8])])
  if kind=='shiftquad':return aff(np.c_[z,z*z,pair(z),np.tanh(z*p[:4]+p[4:8])])
 p0=np.r_[[2.2]*4,[0.]*4, [.4] if kind=='tied' else []]; hi=np.r_[[6.]*4,[2.]*4,[4.] if kind=='tied' else []]; lo=np.r_[[.15]*4,[-2.]*4,[0.] if kind=='tied' else []]
 def fun(p):
  F=f(u,p);return (F@ls(F,v)-v).ravel()
 p=least_squares(fun,p0,bounds=(lo,hi),max_nfev=160).x;c=ls(f(u,p),v);return lambda z:f(z,p)@c,p,c
for kind in ('tanh','tied','sin'):
 fun,p,c=cfit(u,v,kind);print('C1',kind,'params',p,'rms',rms(fun(u)-v),'OOF', [round(rms(cfit(u[jj%4!=k],v[jj%4!=k],kind)[0](u[jj%4==k])-v[jj%4==k]),5) for k in range(4)]);print(c)
for name,make in [('quad',lambda z:aff(np.c_[z,z*z,pair(z)])),('addquad',lambda z:aff(np.c_[z,z*z])),('tri',lambda z:PolynomialFeatures(3).fit_transform(z))]:
 M=make(u); c=ls(M,v2);print('C2',name,'fit',rms(M@c-v2),'OOF',[round(rms(M[jj%4==k]@ls(M[jj%4!=k],v2[jj%4!=k])-v2[jj%4==k]),5) for k in range(4)]);
 if name=='quad':print(c)
x,y,q=[s['train'][k] for k in ('x','y','traj')];t=s['eval_times']; a=calfit(s['panel_a']);p=s['panel_b'];b=ls(aff(p['u']),p['v']);C=cfit(u,v,'tied')[0];c2=lambda z:aff(np.c_[z,z*z,pair(z)])@ls(aff(np.c_[u,u*u,pair(u)]),v2)
spaces={'raw':x,'A+':a[:,0]+a[:,1]*x,'A-':(x-a[:,0])/a[:,1]}
for k,z in list(spaces.items()):spaces[k+'B+']=aff(z)@b;spaces[k+'B-']=(z-b[0])@np.linalg.inv(b[1:]);
def evaluate(M,alpha=.003):
 scores=[]
 for k in range(6):
  va=np.arange(len(x))%6==k;tr=~va
  def f(p):return -np.expm1(-t[:,None]*p)
  def er(p):
   T=f(p);return (T@ls(T,q[tr].T)-q[tr].T).ravel()
  T=f(least_squares(er,[1.8,.4],bounds=(.01,10)).x);Y=np.c_[y[tr],ls(T,q[tr].T).T]; H=Ridge(alpha=alpha).fit(M[tr],Y).predict(M[va]);e=np.c_[H[:,0]-y[va],H[:,1:]@T.T-q[va]];scores.append(np.mean(abs(e),axis=0))
 scores=np.array(scores);return np.r_[scores[:,0].mean(),scores[:,1:].mean(),scores[:,0].max(),scores[:,1:].max()]
rank=[]
for n,z in spaces.items():
 for label,M in [('self',np.c_[z,z*z,pair(z),np.tanh(2.2*z)]),('C1',np.c_[C(z),C(z)**2,pair(C(z))]),('C1+C2',np.c_[C(z),c2(z)]),('self+C1+C2',np.c_[z,z*z,pair(z),C(z),c2(z)]),('C1x',np.c_[z,z*z,pair(z),C(z),z*C(z)]),('C1+mix',np.c_[C(z),C(z)**2,pair(C(z)),c2(z),C(z)*c2(z)])]:
  score=evaluate(M);rank.append((score[:2].sum(),n,label,score))
for row in sorted(rank,key=lambda a:a[0])[:22]: print(row[1:])
