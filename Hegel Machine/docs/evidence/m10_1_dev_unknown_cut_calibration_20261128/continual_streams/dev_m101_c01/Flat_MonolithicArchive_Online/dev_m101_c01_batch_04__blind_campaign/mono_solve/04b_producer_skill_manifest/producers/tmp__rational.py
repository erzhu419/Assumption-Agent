exec(open('/tmp/structure.py').read().split('for k in')[0])
from sklearn.model_selection import KFold
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
F=[(np.arange(56),np.arange(56,72)),*KFold(5,shuffle=True,random_state=12).split(x)]
P=poly(x,2); X=poly(x,1); D=cdist(x,x,'sqeuclidean'); K=P@P.T+np.exp(-D/2)

def structural(i,j,b,kind):
 p=P[i]; w=X[i]; q=b[i]; best=None
 for v in [np.zeros(5)]+[np.eye(5)[k]*.6 for k in range(5)]+[np.random.default_rng(k).normal(0,.5,5) for k in range(5)]:
  def design(p,w,h):
   a=w@h
   if kind=='rat':return p/(1+a*a)[:,None]
   if kind=='ratplus':return np.c_[p,w/(1+a*a)[:,None]]
   if kind=='ratquad':return np.c_[w,p/(1+a*a)[:,None]]
   if kind=='tanh':return np.c_[p,np.tanh(a)]
   if kind=='sin':return np.c_[p,np.sin(a)]
   if kind=='switch':return np.c_[p,np.abs(a)]
  def err(h):
   A=design(p,w,h); return np.r_[A@fit(A,q)-q,.001*h]
  o=least_squares(err,v,bounds=(-8,8),max_nfev=150)
  if best is None or o.cost<best.cost:best=o
 h=best.x; A=design(p,w,h); return design(P[j],X[j],h)@fit(A,q),h,np.sqrt(2*best.cost/len(i))
for kind in ['rat','ratplus','ratquad','tanh','sin','switch']:
 out=[]
 for i,j in F:
  v,h,r=structural(i,j,y,kind); out.append(np.mean(abs(v-y[j])))
 print(kind,np.round(out,5),'last params',h.round(3),r,flush=True)
i,j=F[0]; yh=K[j[:,None],i]@solve(K[i[:,None],i]+.001*np.eye(len(i)),y[i],assume_a='pos');print('readout 56/16 errors by example:'); print(np.c_[x[j],y[j],yh,y[j]-yh].round(4));
r=y-P@fit(P,y); ix=np.argsort(abs(r))[-12:];print('quadratic tails');print(np.c_[ix,x[ix],y[ix],r[ix]].round(3))
