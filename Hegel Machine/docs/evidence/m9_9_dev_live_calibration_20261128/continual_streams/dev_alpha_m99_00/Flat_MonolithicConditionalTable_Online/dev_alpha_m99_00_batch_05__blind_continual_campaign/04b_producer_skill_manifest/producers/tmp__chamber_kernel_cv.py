from chamber_compare import *
from scipy.optimize import least_squares
from itertools import product
np.set_printoptions(precision=6)
# Align the verification panel on its own reference coupons. Deployment uses the prescribed joint train/test panel.
xx=m._features(obs)[good]
score=xx@th[3:]+th[2]
loadcurve=hold[:,None]*(k@th[:2]); loadrate=hold*np.sum(th[:2]/tau)
Y=y-loadcurve; R=r-loadrate
labs=o['lab_id']

def expkernel(t,a):
 return -a*np.expm1(-t/a)

def kernel(name,p,t):
 if name=='linear': return t
 if name=='archive': return -np.expm1(-t[:,None]/np.array([.336621,1.532331]))@np.array([.067513,.984504])+.160197*t
 if name=='half-plus-unit-exp': return .5*(t+expkernel(t,1.))
 if name=='exp': return expkernel(t,p[0])
 if name=='rational': return t/(1+p[0]*t)
 if name=='linear+exp': return p[0]*t+(1-p[0])*expkernel(t,p[1])
 if name=='linear+fixed-exp': return p[0]*t+(1-p[0])*expkernel(t,1.)
 if name=='two-exp': return p[0]*expkernel(t,p[1])+(1-p[0])*expkernel(t,p[2])
 if name=='linear+rational': return p[0]*t+(1-p[0])*t/(1+p[1]*t)
 if name=='material-exp+unit-exp': return p[0]*expkernel(t,tau[1])+(1-p[0])*expkernel(t,1.)

models={'linear':None,'archive':None,'half-plus-unit-exp':None,'exp':([2.0],[.02],[100.]),'rational':([.2],[0.],[100.]),'linear+exp':([.5,1.],[0.,.02],[1.,100.]),'linear+fixed-exp':([.5],[0.],[1.]),'two-exp':([.5,1.,12.],[0.,.02,.02],[1.,100.,100.]),'linear+rational':([.5,.5],[0.,0.],[1.,100.]),'material-exp+unit-exp':([.5],[0.],[1.])}

def fit(name,keep):
 spec=models[name]
 if spec is None: return []
 p,lo,hi=spec
 return least_squares(lambda v:(score[keep,None]*kernel(name,v,t)-Y[keep]).ravel(),p,bounds=(lo,hi),loss='soft_l1',f_scale=.003).x

for name in models:
 e=np.zeros_like(Y)
 for lab0 in np.unique(labs):
  keep=labs!=lab0; p=fit(name,keep)
  e[~keep]=score[~keep,None]*kernel(name,p,t)-Y[~keep]
 p=fit(name,np.ones(len(labs),bool)); print(name, 'full pars',p,'OOF pooled/worst lab MAE',abs(e).mean(),max(abs(e[labs==l]).mean() for l in np.unique(labs)),'worst early/late MAE',max(abs(e[labs==l,:4]).mean() for l in np.unique(labs)),max(abs(e[labs==l,-4:]).mean() for l in np.unique(labs)))
print('rate aligned worst lab MAE',max(np.mean(abs((R-score)[labs==l])) for l in np.unique(labs)))
allx=m._features(obs); _,allhold,_=m._coupon(obs,allx,bridge); allscore=allx@th[3:]+th[2]; ally=obs['strain_curves_obs']-obs['strain_0'][:,None]; allr=obs['instant_rate_obs']
for i in (good,~good):
 p=fit('linear+exp',np.ones(len(labs),bool)); e=allhold[:,None]*(k@th[:2])+allscore[:,None]*kernel('linear+exp',p,t)-ally; q=allhold*np.sum(th[:2]/tau)+allscore-allr; print('rows',sum(i),'curve/rate MAE',abs(e[i]).mean(),abs(q[i]).mean(),'curve/rate bias',e[i].mean(),q[i].mean())
