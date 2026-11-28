from probe import *
import sys
sys.path.insert(0,'/holdout_task'); import batch_solver as bs
u=bs._inverse(x,bs._calibrate(s['panel_a'])); px,py=map(s['panel_b'].get,('u','v')); g=bs._operator(s['panel_b'],u)
from scipy.optimize import minimize_scalar
fun={'rat':lambda q,k:q/(1+k*q*q),'tanh':lambda q,k:np.tanh(k*q),'sin':lambda q,k:np.sin(k*q),'soft':lambda q,k:q/(1+k*q*q)**.5,'cubeRat':lambda q,k:q**3/(1+k*q*q),'atan':lambda q,k:np.arctan(k*q),'erf':__import__('scipy').special.erf}
for name,f in list(fun.items()):
 if name=='erf': f=lambda q,k:__import__('scipy').special.erf(k*q)
 for m in [0,1,2]:
  def fit(k,ret=False):
   A=[np.c_[aff(px),f(px,k)],aff(f(px,k)),np.c_[aff(px),f(px,k),px**3]][m]; w=ls(A,py); e=A@w-py; h=np.sum(np.linalg.qr(A,mode='reduced')[0]**2,axis=1); return (w,e) if ret else np.sqrt(np.mean((e/(1-h[:,None]))**2))
  o=minimize_scalar(fit,bounds=(.03,3.5),method='bounded'); w,e=fit(o.x,True); print(name,m,round(o.x,5),round(o.fun,6),round(np.std(e),6));
  if name=='rat' and m==0: print(w)
A=bs._quadratic(g); Y=np.c_[y,T]; W=ls(A,Y); print('readout stds',np.std(A@W-Y,0)); print('readout coefficients',W[:,[0,1,4,8]]); print('full spatial affine coeff',ls(aff(g),Y)[:,[0,1,4,8]]); print('summary',tr['summary']);
P=np.linalg.svd(T,full_matrices=False)[2][:2].T
basis=[('exp',lambda a:-np.expm1(-ts*a)),('rat',lambda a:ts/(a+ts)),('log',lambda a:np.log1p(ts*a)),('linexp',lambda a:ts*np.exp(-ts*a)),('tanh',lambda a:np.tanh(ts*a)),('pow',lambda a:ts**a),('exp0',lambda a:np.exp(-ts*a))]
res=[]
for k,f in basis:
 for j,h in basis:
  def obj(p):
   B=np.c_[f(np.exp(p[0])),h(np.exp(p[1]))];return (B@ls(B,P)-P).ravel()
  o=least_squares(obj,[0.,-.5],bounds=(-5,4));res.append((np.linalg.norm(o.fun),k,j,np.exp(o.x)))
print('time hypotheses',sorted(res)[:12])
for hi,lo in [(np.arange(52),np.arange(52,72))]+[(a[:52],a[52:]) for a in [np.random.default_rng(j).permutation(72) for j in range(20)]]:
 ss=dict(s,train=dict(tr,x=x[hi],y=y[hi],traj=T[hi]),test={'x':x[lo]}); o=bs.solve_batch(ss); print('split',np.mean(abs(o['y_hat']-y[lo])),np.mean(abs(o['state_pred']-T[lo])),np.mean(abs(o['state_pred']-T[lo]),0).max())
