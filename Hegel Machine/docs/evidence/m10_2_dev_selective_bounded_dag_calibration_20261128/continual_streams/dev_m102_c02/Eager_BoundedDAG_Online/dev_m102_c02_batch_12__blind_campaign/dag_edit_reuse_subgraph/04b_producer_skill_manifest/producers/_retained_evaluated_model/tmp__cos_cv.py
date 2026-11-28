import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from probe import *
from scipy.optimize import minimize_scalar,least_squares
u,v=b['u'],b['v']
def fit(ids):
 def score(w):
  e=res(aff(np.cos(w*u[ids])),v[ids]);return np.sum(e*e)
 grid=np.arange(.25,12.01,.125);k=np.argmin([score(w) for w in grid]);w=minimize_scalar(score,bounds=(max(.01,grid[k]-.125),grid[k]+.125),method='bounded').x
 return w,ls(aff(np.cos(w*u[ids])),v[ids])
w,B=fit(np.arange(len(u)));print('shared cosine frequency and panel RMSE',w,np.sqrt(np.mean((aff(np.cos(w*u))@B-v)**2)))
panel_err=[]
for j in range(len(u)):
 wj,Bj=fit(np.arange(len(u))[np.arange(len(u))!=j]);panel_err.extend(np.abs((aff(np.cos(wj*u[j:j+1]))@Bj-v[j]).ravel()))
print('leave-one-reference-out max and mean abs',max(panel_err),np.mean(panel_err))
f=aff(np.cos(w*(T['x']-o)/g))@B;H=aff(f);G=np.c_[H,np.sum(f[:,:2]*f[:,2:],axis=1)];ts=S['eval_times'];N=len(f)
def modes(log_rates):return -np.expm1(-np.exp(log_rates[:,None])*ts)
def temporal(beta):
 def err(log_rates):
  m=modes(log_rates);return (beta@np.linalg.pinv(m)@m-beta).ravel()
 z=least_squares(err,np.log([.25,1.4]),bounds=(np.log([.01,.03]),np.log([5,20])),max_nfev=300);return z,modes(z.x)
print('relaxation rates on full public train',np.exp(temporal(ls(H,T['traj']))[0].x))
r=[]
for trial in range(41):
 j=np.arange(N) if trial==0 else np.random.default_rng(trial).permutation(N);n=52 if trial==0 else 44;i,j=j[:n],j[n:];beta=ls(H[i],T['traj'][i]);_,_,R=np.linalg.svd(beta,full_matrices=False);m=temporal(beta)[1];yerr=np.abs(G[j]@ls(G[i],T['y'][i])-T['y'][j]);ps=[H[j]@beta,H[j]@beta@R[:2].T@R[:2],H[j]@beta@np.linalg.pinv(m)@m];a=[yerr.mean()]+[np.abs(p-T['traj'][j]).mean() for p in ps];r.append(a)
print('official proxy, worst randomized split (scalar / direct / SVD / relaxation)',r[0],np.max(r[1:],axis=0))
