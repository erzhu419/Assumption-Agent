from discover04 import *
from scipy.linalg import solve
from scipy.spatial.distance import cdist
A,B,C,D=fit_all(s); x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; T=np.c_[y,z]; t=s['eval_times']; rng=np.random.default_rng(91); cv=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in (rng.permutation(72) for _ in range(7))]; print('B',B.round(4));
def mult(f,g):return (f[:,:,None]*g[:,None,:]).reshape(len(f),-1)
def errors(p,j):
 e=abs(p-T[j]);return [e[:,0].mean(),e[:,1:].mean()]
rows={}
for name,u in [('raw',x),('B',aff(x)@B),('iB',aff((x-A[0])/A[1])@B),('aB',aff(A[0]+x*A[1])@B)]:
 Q=poly(u,2); P=poly(u,3)[:,15:]; H=C(u); J=D(u); d=cdist(u,u,'sqeuclidean');
 for nm,F in [('baseline',np.c_[Q,J,mult(u,J)]),('cubic',np.c_[Q,P]),('odd',np.c_[Q,P,H,mult(u,H)]),('both',np.c_[Q,P,H,J,mult(u,np.c_[H,J])])]:
  for pen in [.03,.12,.24,.5]:
   for k in [0,.3,1,3]:
    K=F@F.T+(k*np.exp(-.5*d/(1.0 if k<1 else 2.0)**2)); rr=[]
    for i,j in cv:
     w=solve(K[np.ix_(i,i)]+pen**2*np.eye(len(i)),T[i],assume_a='pos');rr.append(errors(K[np.ix_(j,i)]@w,j))
    rows[name,nm,pen,k]=np.array(rr)
 # exploit coordinatewise nonlinearities and explicit sums/differences rather than a generic cube
 W=np.c_[u,u[:,[0,0,0,1,1,2]]+u[:,[1,2,3,2,3,3]],u[:,[0,0,0,1,1,2]]-u[:,[1,2,3,2,3,3]], x]
 for label,g in [('abs',np.abs(W)),('odd',W*abs(W)),('saturate',W/(1+abs(W))),('even',np.sqrt(1+W*W)-1),('exp',np.sign(W)*(-np.expm1(-1.8*abs(W)))),('trig',np.sin(W))]:
  for F in [np.c_[Q,g,mult(u,g[:,:4])],np.c_[Q,P,g,mult(u,g[:,:4])]]:
   for pen in [.02,.08,.2,.4]:
    K=F@F.T; rr=[]
    for i,j in cv:
     w=solve(K[np.ix_(i,i)]+pen**2*np.eye(len(i)),T[i],assume_a='pos');rr.append(errors(K[np.ix_(j,i)]@w,j))
    rows[name,label,F.shape[1],pen]=np.array(rr)
for c in range(2):
 for key in [lambda e:e.mean(),lambda e:e.max()]:
  print('target',c)
  for nm,e in sorted(rows.items(),key=lambda ne:key(ne[1][:,c]))[:6]:print(nm,e[:,c].round(5),'mean/worst',e[:,c].mean().round(5),e[:,c].max().round(5))
print('time rank1',np.linalg.svd(z,full_matrices=False)[2][0]); w=np.linalg.lstsq(poly(x,2),T,rcond=None)[0]; print('quadratic',w[:,[0,-1]].round(3)); print('resid corr',np.corrcoef((T-poly(x,2)@w).T)[0,1:])
