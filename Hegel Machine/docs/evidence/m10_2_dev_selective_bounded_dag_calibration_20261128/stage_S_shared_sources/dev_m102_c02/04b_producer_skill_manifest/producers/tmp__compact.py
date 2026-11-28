import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import _affine as A,_quadratic as Q
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,z=(s['train'][k] for k in ('x','y','traj')); t=s['eval_times']; ids=np.arange(len(x)); rng=np.random.default_rng(1); slices=[ids[52:],*np.array_split(rng.permutation(len(x)),4)]
for col in x.T:
 o=np.argsort(col); slices.extend([o[:14],o[-14:]])
slices.append(np.argsort(np.linalg.norm(x-x.mean(0),axis=1))[-14:]); splits=[(np.setdiff1d(ids,j),j) for j in slices]
a=s['panel_a']; u,v=a['u'],a['v']; aa=[]
for k in range(4):
 keep=np.abs(v[:,k]-u[:,k])<2; aa.append(np.linalg.lstsq(A(u[keep,k:k+1]),v[keep,k],rcond=None)[0]);
b=s['panel_b']; B=np.linalg.lstsq(A(b['u']),b['v'],rcond=None)[0]; ao,ag=np.array(aa).T; un=(x-ao)/ag; C={'raw':A(x)@B,'cal':A(x*ag+ao)@B,'uncal':A(un)@B}; rms=lambda e: np.mean(e**2)**.5

def evaluate(name,fn,target=y):
 es=[abs(fn(i,j)-target[j]) for i,j in splits]; m=np.array([e.mean() for e in es]); print(name,'design/mean/worst',np.round([m[0],m.mean(),m.max()],6),'worst horizon',round(max(e.mean(0).max() for e in es),6))
def predict(F,target,i,j): return F[j]@np.linalg.lstsq(F[i],target[i],rcond=None)[0]
evaluate('full quadratic',lambda i,j:predict(Q(x),y,i,j))
for name,c in C.items():
 p=c[:,:2]*c[:,2:]; h=p.sum(1); F=np.c_[A(c),h]; coef=np.linalg.lstsq(F,y,rcond=None)[0]; print(name,'compact coef',coef,'res RMS',rms(y-F@coef)); evaluate(name+' cross-sum',lambda i,j:predict(F,y,i,j)); F2=np.c_[A(c),p]; evaluate(name+' cross-pair',lambda i,j:predict(F2,y,i,j));
 for tag,q in [('cross',h),('energy',np.linalg.norm(c,axis=1)),('pairdiff',p[:,0]-p[:,1]),*[(f'c{k}',c[:,k]) for k in range(4)]]:
  def fit(i,j):
   def design(a,k): return np.c_[A(c[k]),h[k]/(1+a*q[k]**2)]
   def resid(a):
    G=design(a[0],i); return G@np.linalg.lstsq(G,y[i],rcond=None)[0]-y[i]
   a=least_squares(resid,[.02],bounds=(0,2),max_nfev=40).x[0]; return design(a,j)@np.linalg.lstsq(design(a,i),y[i],rcond=None)[0]
  evaluate(name+' rational '+tag,fit)
X=A(x)
def basis(p): return -np.expm1(-np.exp(p[:,None])*t)
def temporal(F,i,j):
 G=F[i]; Z=F[j]; W=np.linalg.lstsq(G,z[i],rcond=None)[0]
 def resid(p):
  V=basis(p); return ((G@W)@np.linalg.pinv(V)@V-G@W).ravel()
 p=least_squares(resid,np.log([.4,1.8]),bounds=(np.log([.04,.7]),np.log([1.3,6])),max_nfev=60).x; V=basis(p); D=W@np.linalg.pinv(V); return Z@D@V,p,D
for name,c in {'x':x,**C}.items():
 F=A(c); pred,p,D=temporal(F,ids,ids); print(name,'exp rates',np.exp(p),'amplitudes\n',D); evaluate(name+' two relaxation modes',lambda i,j:temporal(F,i,j)[0],z)
