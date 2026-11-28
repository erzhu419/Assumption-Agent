import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_c']['u']; L=np.c_[np.ones(len(x)),x]
rng=np.random.default_rng(1)
Q=np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
for key in ('v1','v2'):
 y=s['panel_c'][key]; print('\n',key)
 b=np.linalg.lstsq(np.c_[L,Q],y,rcond=None)[0]; mats=np.zeros((4,4,4));
 for c,(i,j) in zip(b[5:],itertools.combinations_with_replacement(range(4),2)):
  mats[:,i,j]=mats[:,j,i]=c/(1+(i!=j))
 print('quadratic eigenvalues',np.round(np.linalg.eigvalsh(mats),3)); print('quadratic diagonals',np.round(np.diagonal(mats,axis1=1,axis2=2),3))
 dirs=np.concatenate((np.eye(4),[np.eye(4)[i]+sign*np.eye(4)[j] for i,j in itertools.combinations(range(4),2) for sign in [1,-1]])); z=x@dirs.T
 def rmse(f):
  f=np.c_[L,f]; u,d,_=np.linalg.svd(f,full_matrices=False); u=u[:,d>d[0]*1e-10]; r=y-u@(u.T@y); h=(u*u).sum(1); return np.round(np.sqrt((r*r).mean(0)),5), np.round(np.sqrt(((r/np.maximum(1-h,1e-14)[:,None])**2).mean(0)),5)
 for name,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('square',np.square),('cubic',lambda z:z**3),('rational',lambda z:z/(1+z*z)),('gauss',lambda z:np.exp(-z*z))]:
  for scale in [0.5,1,1.5,2,3]:
   if name in ('square','cubic') and scale!=1: continue
   print('axis',name,scale,rmse(fun(scale*x)))
 for j in range(4):
  results=[]
  for name,fun in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rational',lambda z:z/(1+z*z)),('gauss',lambda z:np.exp(-z*z))]:
   # Single projection with bias and affine nuisance; also fit without affine nuisance.
   for use_affine in [False,True]:
    base=L if use_affine else L[:,:1]; qb=np.linalg.qr(base)[0]; target=y[:,j]-qb@(qb.T@y[:,j]);
    def err(w):
     col=fun(x@w[:4]+w[4]); col-=qb@(qb.T@col); return target-col*((col@target)/max(col@col,1e-30))
    ev,vec=np.linalg.eigh(mats[j]); starts=[np.r_[v,k] for v in [b[1:5,j], 1.8*vec[:,np.argmax(abs(ev))],*rng.normal(0,.8,(5,4))] for k in [0.,.7]]
    fits=[least_squares(err,np.clip(w,-5.9,5.9),bounds=(-6,6),max_nfev=180) for w in starts]; opt=min(fits,key=lambda fit:fit.cost)
    results.append((round(np.sqrt(np.mean(opt.fun**2)),6),name,use_affine,np.round(opt.x,3).tolist()))
  print('single',j,sorted(results))
print('traj svd',np.linalg.svd(s['train']['traj'],compute_uv=False)); print('x and B',np.round(np.linalg.svd(np.linalg.lstsq(np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'],rcond=None)[0][1:],compute_uv=False),5))
