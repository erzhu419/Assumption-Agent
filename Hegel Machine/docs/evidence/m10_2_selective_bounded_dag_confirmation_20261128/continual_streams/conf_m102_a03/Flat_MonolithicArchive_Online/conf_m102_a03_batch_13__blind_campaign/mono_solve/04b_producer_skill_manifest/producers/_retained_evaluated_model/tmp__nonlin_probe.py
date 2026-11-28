exec(open('/tmp/readout_probe.py').read().split("print('mix'")[0])
from itertools import combinations,combinations_with_replacement
A=aff(w); p=w[:,0]*w[:,2]+w[:,1]*w[:,3]; B=np.c_[A,p]; ii=np.arange(52); jj=np.arange(52,72)
lin=fit(B[ii],y[ii]); print('base coef',lin)
def score(pr):return np.mean(abs(pr[ii]-y[ii])),np.mean(abs(pr[jj]-y[jj]))
models=[]
def var(name,design,starts):
 def pred(q):
  a=design(q);return a@fit(a[ii],y[ii])
 for q in starts:
  with np.errstate(all='ignore'):
   r=least_squares(lambda q:(pred(q)-y)[ii],q,max_nfev=350)
  models.append((*score(pred(r.x)),name,r.x,fit(design(r.x)[ii],y[ii])))
for name,x in [('w'+str(i),w[:,i]) for i in range(4)]+[('pair',p),('norm',np.linalg.norm(w,axis=1)),('index',A@lin[:5])]+[(f'w{i}{sg:+}w{j}',w[:,i]+sg*w[:,j]) for i,j in combinations(range(4),2) for sg in [-1,1]]:
 for scope in [0,1,2]:
  def des(q):
   d=1+q[0]**2*x*x
   return [np.c_[A,p/d],np.c_[A/d[:,None],p], B/d[:,None]][scope]
  var(f'rat {scope} {name}',des,[[.45],[1.25]])
 for label,f in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan)]:
  var(f'add {label} {name}',lambda q:np.c_[B,f(q[0]*x+q[1])],[[1.25,0],[2.15,0]])
for label,f in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('cube',lambda x:x**3),('rational',lambda x:x/(1+x*x))]:
 for prior in [0,1]:
  def des(q):return np.c_[B, f(A@q[:5]+prior*p*q[5])]
  var('ridge '+label+str(prior),des,[np.r_[0,k*lin[1:5],k*lin[5]] for k in [.4,1.25,2.15]])
for label,f in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('cube',lambda x:x**3),('rational',lambda x:x/(1+x*x))]:
 var('sep '+label,lambda q:np.c_[B,f(w*q[0]+q[1])],[[1.25,0],[2.15,0]])
for a,b,n,q,c in sorted(models,key=lambda r:r[0])[:20]: print('fit',round(a,6),round(b,6),n,np.round(q,4),np.round(c,4))
for a,b,n,q,c in sorted(models,key=lambda r:r[1])[:8]: print('cv',round(a,6),round(b,6),n,np.round(q,4))
R=tr['traj']-A@fit(A[ii],tr['traj'][ii]); print('traj singular residual',np.linalg.svd(R,compute_uv=False)); print('traj coefs',fit(A[ii],tr['traj'][ii]));
