import pickle,runpy,itertools
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; old=runpy.run_path('/tmp/hegel_continual_archive/dev_m101_c04_batch_00_solver.py'); np.set_printoptions(precision=4,suppress=True,linewidth=145)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def stats(e):return np.mean(abs(e)),np.sqrt(np.mean(e**2))
tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; cal=old['_calibration'](s); q=(x-cal[0])/cal[1]; z=old['_coordinates'](x,cal); c=old['_cross'](z); base=np.c_[z,c]; res=y-base@ls(base,y); tv=s['eval_times']; rng=np.random.default_rng(3187); folds=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in [rng.permutation(72) for _ in range(6)]]
def cv(X):return np.array([np.mean(abs(X[b]@ls(X[a],y[a])-y[b])) for a,b in folds])
def report(n,X):
 e=cv(X); print(n,round(e[0],5),round(e.mean(),5),round(e.max(),5),tuple(round(i,5) for i in stats(X@ls(X,y)-y)))
print('base coeff',ls(base,y), 'base residual quantiles',np.quantile(res,[0,.1,.25,.5,.75,.9,1])); print('error vs state:',ls(np.c_[base,T],y)[5:]); report('base',base)
for name,u in [('raw',x),('calibrated',q),('mixed',z)]:
 a=aff(u); pairs=np.column_stack([u[:,i]*u[:,j] for i,j in itertools.combinations(range(4),2)]); norm=np.sum(u**2,1); k=np.c_[u,u[:,[0]]+u[:,[1]],u[:,[0]]-u[:,[1]],u[:,[0]]+u[:,[2]],u[:,[0]]-u[:,[2]],u[:,[0]]+u[:,[3]],u[:,[0]]-u[:,[3]],u[:,[1]]+u[:,[2]],u[:,[1]]-u[:,[2]],u[:,[1]]+u[:,[3]],u[:,[1]]-u[:,[3]],u[:,[2]]+u[:,[3]],u[:,[2]]-u[:,[3]],c,norm]; print(name)
 for nn,F in [('cubes',u**3),('threeway',np.column_stack([np.prod(u[:,i],1) for i in itertools.combinations(range(4),3)])),('sin',np.sin(1.7*u)),('sin2',np.sin(3.4*u)),('cos',np.cos(1.7*u)),('rational',u/(1+1.7*u**2)),('quad',np.c_[u**2,pairs]),('mod-cross',np.c_[c[:,None]*u,c[:,None]*u**2]),('gate-cross',c[:,None]/(1+1.7*u**2)),('gate-linear',u/(1+norm[:,None]*1.7))]:report(nn,np.c_[base,F])
 dic={}
 for j,v in enumerate(k.T):
  dic.update({(j,n):f for n,f in [('square',v*v),('cube',v**3),('sin',np.sin(1.7*v)),('cos',np.cos(1.7*v)),('sin2',np.sin(3.4*v)),('cos2',np.cos(3.4*v)),('rat',v/(1+1.7*v*v)),('exp',np.exp(-1.7*v*v)),('abs',abs(v)),('c_gate',c/(1+1.7*v*v)),('c_prod',c*v)]})
 best=sorted((cv(np.c_[base,v]).mean(),str(n),v) for n,v in dic.items())[:5]; print('best single',[(round(i,5),n) for i,n,_ in best]);
for key in ['v1','v2']:
 p=s['panel_c']; u=p['u']; V=p[key]; print('C',key)
 for n,fn in [('quadratic',lambda u,k:u*u),('cos',lambda u,k:np.cos(k*u)-1),('sin',lambda u,k:np.sin(k*u)),('tanh',lambda u,k:np.tanh(k*u)),('rational',lambda u,k:u*u/(1+k*u*u)),('cos+sin',lambda u,k:np.c_[np.cos(k*u)-1,np.sin(k*u)])]:
  def err(k):
   X=np.c_[aff(u),fn(u,k)]; return np.mean((X@ls(X,V)-V)**2)
  r=minimize_scalar(err,bounds=(.1,5),method='bounded'); k=r.x; U=np.c_[aff(u),fn(u,k)]; W=ls(U,V); F=np.c_[aff(z),fn(z,k)]@W; print(n,round(k,4),round(r.fun**.5,6), 'w0',W[0]); report('C augmented',np.c_[base,F]); report('C replace',np.c_[F,c])
print('trajectory')
for d in (z,aff(z)):
 w=ls(d,T); print('coeff',w); print('singular',np.linalg.svd(w,compute_uv=False));
 for a,b in folds[:2]:
  w0=ls(d[a],T[a]); w1=old['_relaxation'](d[a],T[a],tv); print('direct/modal',stats(T[b]-d[b]@w0),stats(T[b]-d[b]@w1))
