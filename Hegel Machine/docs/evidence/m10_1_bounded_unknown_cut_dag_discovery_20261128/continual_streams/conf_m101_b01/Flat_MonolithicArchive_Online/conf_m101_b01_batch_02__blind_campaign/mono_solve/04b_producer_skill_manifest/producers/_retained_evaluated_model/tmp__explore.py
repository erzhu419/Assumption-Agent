import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; n=len(x)
np.set_printoptions(precision=5,suppress=True,linewidth=160)
def aff(z):return np.c_[np.ones(len(z)),z]
def ls(d,v):return np.linalg.lstsq(d,v,rcond=None)[0]
def f(p):
 u,v=p['u'],p['v']; rng=np.random.default_rng(23); best=None
 for ii in range(150):
  j=rng.choice(len(u),5,replace=False); w=ls(aff(u[j]),v[j]); e=np.linalg.norm(aff(u)@w-v,axis=1); score=np.median(e)
  if best is None or score<best[0]:best=score,e
 keep=best[1]<max(0.001,5*best[0]);w=ls(aff(u[keep]),v[keep]);return w,np.linalg.norm(aff(u)@w-v,axis=1)
wa,ea=f(s['panel_a']); print('panel A',wa,'errors sorted',np.sort(ea)); wb=ls(aff(s['panel_b']['u']),s['panel_b']['v']); print('ranges',x.min(0),x.max(0),s['test']['x'].min(0),s['test']['x'].max(0))
def b(r): return -np.expm1(-np.exp(r[:,None])*t)
fit=least_squares(lambda r:(ls(b(r).T,tr.T).T@b(r)-tr).ravel(),[-1.,1.],bounds=(-8,5)); B=b(fit.x); A=ls(B.T,tr.T).T; print('rates',np.exp(fit.x),'temporal rmse',np.mean((A@B-tr)**2)**.5,'mode mean/std',A.mean(0),A.std(0)); print('linear relationship y -> modes',np.corrcoef(np.c_[y,A].T)); tar=np.c_[y,A]
z=(x-wa[0])@np.linalg.inv(wa[1:]); coords={'raw':x,'A inverse':z,'A forward':aff(x)@wa,'B forward':aff(x)@wb,'AB forward':aff(z)@wb,'AB direct':aff(aff(x)@wa)@wb,'AB inverse':(z-wb[0])@np.linalg.pinv(wb[1:])}
def poly(z,k,mix=True):
 return np.column_stack([np.ones(len(z))]+[np.prod(z[:,j],axis=1) for d in range(1,k+1) for j in combinations_with_replacement(range(z.shape[1]),d) if mix or j[0]==j[-1]])
def test(d):
 p=d[52:]@ls(d[:52],tar[:52]); return np.r_[np.mean(abs(p[:,0]-y[52:])), np.mean(abs(p[:,1:]@B-tr[52:])), np.mean(abs(p[:,1:]-A[52:]),0)]
for name,z0 in coords.items():
 print(name, 'std',z0.std(0)); print('linear', test(aff(z0)), 'diag quadratic',test(poly(z0,2,False)),'full quadratic',test(poly(z0,2)),'diag cubic',test(poly(z0,3,False)),'full cubic',test(poly(z0,3)))
 for name2,g in [('tanh',np.tanh),('atan',np.arctan),('asinh',np.arcsinh),('rational',lambda a:a/(1+a*a)**.5)]:
  print(name2,[(c,test(poly(g(c*z0),2))) for c in (.5,1.,2.)]);
