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
from scipy.optimize import minimize_scalar
c=s['panel_c']; u=c['u']; v=c['v2']; i,j=np.array([(0,1),(0,3),(1,2),(2,3)]).T
mix=lambda a:a[:,i]*a[:,j]
z=coords['AB forward']; W=ls(aff(mix(u)),c['v1']); q=aff(mix(z))@W
p=q[:,[0,1]]*q[:,[2,3]]
ix=np.arange(72); order=np.argsort(np.linalg.norm(z,axis=1)); folds=[(ix[:52],ix[52:]),(order[:56],order[56:])]+[(np.setdiff1d(ix,b),b) for b in np.array_split(np.random.default_rng(234).permutation(72),5)]
def score(name, design):
 e=[]; w=[]
 for a,b in folds:
  d=design(a); w.append(ls(d[a],y[a])); e.append(np.mean(abs(d[b]@w[-1]-y[b])))
 d=design(ix); r=y-d@ls(d,y); print(name, '52/20, tail, worst random, fit',np.r_[e[:2],max(e[2:]),np.mean(abs(r))]); return r
r=score('full q quadratic',lambda a:poly(q,2)); print('y residual largest',np.sort(abs(r))[-10:]); rr=score('two q crosses',lambda a:np.c_[aff(q),p]); print('two-cross weights',ls(np.c_[aff(q),p],y)); print('two-cross residual largest',np.sort(abs(rr))[-10:]); print('correlation with q cubic',np.corrcoef(np.c_[rr,poly(q,3)[:,15:]].T)[0,1:]);
for name,fun in [('cos',lambda x,k:np.cos(np.sqrt(k)*x)),('rational',lambda x,k:1/(1+k*x*x)),('exp',lambda x,k:np.exp(-k*x*x)),('sqrt',lambda x,k:(1+k*x*x)**-.5),('log',lambda x,k:np.log1p(k*x*x)),('quartic',lambda x,k:x*x+k*x**4)]:
 def d(x,k): return np.c_[aff(x),fun(x,k)]
 def cost(k):
  D=d(u,k); return np.mean((D@ls(D,v)-v)**2)
 fit=minimize_scalar(cost,bounds=(.002,3),method='bounded'); k=fit.x; D=d(u,k); w=ls(D,v); print('c2',name,k,fit.fun**.5); g=d(z,k)@w; h=d(q,k)@w
 for tag,F in [('q+v2(z)',np.c_[aff(q),g]),('q+p+v2(z)',np.c_[aff(q),p,g]),('q+p+v2(q)',np.c_[aff(q),p,h])]:
  score(tag,lambda a:F)
for name,fun in [('sin',lambda x,k:np.sin(k*x)/k),('tanh',lambda x,k:np.tanh(k*x)/k),('rational',lambda x,k:x/(1+k*x*x)),('atan',lambda x,k:np.arctan(k*x)/k)]:
 for mode in ('cross','main','both'):
  def design(a):
   def d(k):return np.c_[aff(q) if mode=='cross' else aff(fun(q,k)),p if mode=='main' else fun(p,k)]
   def loss(k):
    D=d(k)[a]; return np.mean((D@ls(D,y[a])-y[a])**2)
   return d(minimize_scalar(loss,bounds=(.001,1.6),method='bounded').x)
  score(name+' '+mode,design)
