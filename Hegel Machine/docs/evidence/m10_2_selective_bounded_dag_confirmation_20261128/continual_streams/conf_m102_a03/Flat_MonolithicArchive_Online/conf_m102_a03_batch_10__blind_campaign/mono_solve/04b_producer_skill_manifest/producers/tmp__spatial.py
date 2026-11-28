import numpy as np, pickle, runpy
from scipy.linalg import solve
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr
m=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_a03_batch_09_solver.py'); p=runpy.run_path('/tmp/panels.py'); s=p['s']; fit=p['fit']; aff=p['aff']; d=s['train']; Y=np.c_[d['y'],d['traj']]; A=m['_calibration'](s['panel_a']); z=m['_decode'](d['x'],A); print('calibration',A,'z ranges',np.min(z,axis=0),np.max(z,axis=0));
R={}; F={}
for k,l,names in [('panel_b','v',['dead_pre']),('panel_c','v1',['softsign_pre','softsign_add']),('panel_c','v2',['xcos_pre','xcos_add'])]:
 u=s[k]['u']; v=s[k][l];
 for cv,e,n,q,fn,f in p['get_models'](u,v):
  if n not in names: continue
  coef=fit(f(u,q),v); label=l+n[-3:]; R[label]=fn(z); F[label]=f(z,q)[:,1:]; print(label,q,'coeff',np.round(coef,4))
  for side in [coef[5:]]:
   if len(side):print('extra/main ratios',np.round(side/coef[1:5],3))
a=R['v1pre']; b=R['v2pre']; h=R['vpre']; shift=lambda v:np.roll(v,-1,axis=1)
G={'z':z,**R,**{'F'+k:v for k,v in F.items()},'ab':np.c_[a,b],'ab+z':np.c_[a,b,z],'ab+h':np.c_[a,b,h],'a+h':np.c_[a,h],'b+h':np.c_[b,h],'ab+a*b':np.c_[a,b,a*b],'ab+a*broll':np.c_[a,b,a*shift(b)],'a*b':a*b,'a*broll':a*shift(b),'za':np.c_[z,a,z*a],'zb':np.c_[z,b,z*b],'ab+all':np.c_[a,b,z,h]}
S=[(np.arange(52),np.arange(52,72))]+[(np.random.default_rng(i).permutation(72)[:52],np.random.default_rng(i).permutation(72)[52:]) for i in range(3)]
def norms(X,ix):return X/np.maximum(X[ix].std(0),.01)
def evaluate(X,ix,jx,lam):
 X=aff(norms(X,ix)); xt=X[ix]; yh=X[jx]@xt.T@solve(xt@xt.T+lam*np.eye(len(ix)),Y[ix],assume_a='pos'); e=np.mean(abs(yh-Y[jx]),axis=0); return e[0],e[1:].mean(),e[-1]
for k, g in G.items():
 models=[]
 for deg in (1,2,3):
  if g.shape[1]>12 and deg==3:continue
  Q=np.array([np.prod(g[:,ij],axis=1) for n in range(1,deg+1) for ij in cwr(range(g.shape[1]),n)]).T
  for lam in (.000003,.001,.02,.2,2.):
   V=np.array([evaluate(Q,i,j,lam) for i,j in S]); models.append((np.max(V[:,:2]),deg,lam,V))
 mod=min(models,key=lambda g:g[0]); print(k,mod[:3],np.round(mod[3],4).tolist(),flush=True)
