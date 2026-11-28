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
c=s['panel_c']; u=c['u']; np.set_printoptions(precision=4,suppress=True)
for key in ('v1','v2'):
 v=c[key]; print('\nC',key)
 for name,d in [('qdiag',poly(u,2,False)),('qfull',poly(u,2)),('cdiag',poly(u,3,False)),('q+sincos',np.c_[poly(u,2),np.sin(u),np.cos(u)]),('tanh',np.c_[aff(u),np.tanh(u)]),('atan',np.c_[aff(u),np.arctan(u)]),('sin',np.c_[aff(u),np.sin(u)]),('rational',np.c_[aff(u),u/(1+u*u)])]:
  w=ls(d,v); h=np.sum((d@np.linalg.pinv(d))*np.eye(len(d)),axis=1); print(name,'rmse',np.sqrt(np.mean((d@w-v)**2)),'loo mae',np.mean(abs(d@w-v)/np.maximum(1-h[:,None],1e-9)))
  if name=='qfull': print(w)
for name,z0 in coords.items():
 d=poly(z0,2); w=ls(d[:52],tar[:52]); print('quad',name,'coeffs',w if name in ['raw','AB forward'] else '', 'train mae', np.mean(abs((d[:52]@w-tar[:52])),0));
 for key in ('v1','v2'):
  for k in (1,2):
   dc=poly(u,k); q=poly(z0,k)@ls(dc,c[key]); print('C composition',name,key,k,test(aff(q)),test(poly(q,2)))
