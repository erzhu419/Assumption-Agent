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
from scipy.special import erf
c=s['panel_c']; u=c['u']; v=c['v2']; i,j=np.array([(0,1),(0,3),(1,2),(2,3)]).T
mix=lambda a:a[:,i]*a[:,j]
z=coords['AB forward']; W=ls(aff(mix(u)),c['v1']); q=aff(mix(z))@W
print('sparse c1 panel RMSE',np.std(aff(mix(u))@W-c['v1'],0)); print('q ranges',q.min(0),q.max(0)); print('quadratic q scalar coeffs',ls(poly(q[:52],2),y[:52])); print('traj direct q fit RMSE',np.std(aff(q)@ls(aff(q),tr)-tr,0))
for name,fun in [('sin',np.sin),('tanh',np.tanh),('erf',erf),('atan',np.arctan),('cos',np.cos),('exp',np.exp),('rational',lambda a:1/(1+a*a))]:
 for base in [False,True]:
  def d(a,p):return np.c_[aff(a) if base else np.ones(len(a)),fun(p[0]*a+p[1:])]
  fit=least_squares(lambda p:(d(u,p)@ls(d(u,p),v)-v).ravel(),[1.,.2,.2,.2,.2],bounds=([.08,-4,-4,-4,-4],[6,4,4,4,4]),max_nfev=100)
  print('v2',name,base,np.std(fit.fun),fit.x)
folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(234).permutation(72),5)]
def cv(d):
 scores=[np.mean(abs(d[b]@ls(d[a],y[a])-y[b])) for a,b in folds]; return np.r_[scores[0],np.mean(scores[1:]),np.max(scores[1:])]
for name,d in [('q affine',aff(q)),('q pure curvature',np.c_[aff(q),q*q]),('q quad',poly(q,2)),('q diag cubic',poly(q,3,False)),('q quad+diag cubic',np.c_[poly(q,2),q**3]),('q all cubic',poly(q,3)),('z quad',poly(z,2)),('q quad plus z',np.c_[poly(q,2),z]),('q quad plus square z',np.c_[poly(q,2),z*z])]:
 print('scalar',name,cv(d),'fit MAE',np.mean(abs(d@ls(d,y)-y)))
for fun in [np.sin,np.tanh,np.arctan]:
 for scale in [.25,.5,1.,2.]:
  print('scalar transform',fun.__name__,scale,cv(aff(fun(scale*q))),cv(poly(fun(scale*q),2)),cv(np.c_[aff(q),fun(scale*q)]))
