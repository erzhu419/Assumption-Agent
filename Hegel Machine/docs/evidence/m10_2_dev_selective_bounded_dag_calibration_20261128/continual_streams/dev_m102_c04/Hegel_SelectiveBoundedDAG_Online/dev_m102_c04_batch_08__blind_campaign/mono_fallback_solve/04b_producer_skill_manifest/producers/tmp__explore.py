import pickle,numpy as np
from scipy.optimize import least_squares,minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
a=s['panel_a']; u,v=a['u'],a['v']; m=np.linalg.norm(u-v,axis=1)<1
w=ls(aff(u[m]),v[m]); print('cal',m.sum(),w, 'rmse',np.std(aff(u[m])@w-v[m]));
ops={}
for p,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][j]; print('\n',p,j,'linear residual',np.std(aff(u)@ls(aff(u),v)-v))
 results=[]
 for name,fun in [('soft',lambda q,k:q/(1+k*abs(q))),('ratsq',lambda q,k:q*q/(1+k*q*q)),('rat',lambda q,k:q/(1+k*q*q)),('tanh',lambda q,k:np.tanh(k*q)),('sin',lambda q,k:np.sin(k*q)),('cos',lambda q,k:np.cos(k*q)),('square',lambda q,k:q*q),('abs',lambda q,k:abs(q)),('poly2',lambda q,k:PolynomialFeatures(2,include_bias=False).fit_transform(q)[:,4:])]:
  for mode in [0,1]:
   def design(x,k,mode=mode,fun=fun):return aff(fun(x,k)) if not mode else np.c_[aff(x),fun(x,k)]
   def err(k):
    f=design(u,k); return np.std(f@ls(f,v)-v)
   o=minimize_scalar(err,bounds=(.005,4),method='bounded'); f=design(u,o.x); coef=ls(f,v); results.append((o.fun,name,mode,o.x,lambda x,d=design,k=o.x,c=coef:d(x,k)@c,coef))
 results.sort(key=lambda r:r[0]); print([r[:4] for r in results[:8]]); print(results[0][-1]); ops[j]=results[0][4]
x=s['train']['x']; z=aff(x)@w; y=s['train']['y']; t=s['eval_times']; tr=s['train']['traj']; Y=np.c_[y,tr]; print('svd',np.linalg.svd(tr,compute_uv=False)); print('svd incl y',np.linalg.svd(Y,compute_uv=False)); print('y by tr rmse',np.std(aff(tr)@ls(aff(tr),y)-y));
for kind in range(4):
 def tm(k):
  e=-np.expm1(-t[:,None]*k)
  return [e,np.c_[e,t],np.c_[e,np.ones(len(t))],np.c_[e,t,np.ones(len(t))]][kind]
 for n in [1,2,3]:
  def resid(k):
   d=tm(k); return (d@ls(d,tr[:52].T)-tr[:52].T).ravel()
  o=least_squares(resid,np.geomspace(.3,3,n),bounds=(.001,30),max_nfev=150); print('time',kind,n,o.x,np.std(o.fun))
for nm,X in [('raw',x),('cal',z),*((k,f(z)) for k,f in ops.items()),('all',np.column_stack([z]+[f(z) for f in ops.values()])),('b,c',np.column_stack([f(z) for f in ops.values()]))]:
 print('\nfeat',nm)
 for d in range(1,5):
  f=PolynomialFeatures(d).fit_transform(X)
  if f.shape[1]>3000:break
  for reg in [1e-5,.01,.1,1]:
   K=f@f.T; pr=K[52:,:52]@np.linalg.solve(K[:52,:52]+reg*np.eye(52),Y[:52]); print(d,reg,np.mean(abs(pr-Y[52:]),axis=0)[[0,1,4,8]].round(4),end='; ')
  print()
 D=cdist(X,X,'sqeuclidean');
 for g in [.03,.1,.3,1]:
  K=np.exp(-g*D); pr=K[52:,:52]@np.linalg.solve(K[:52,:52]+.001*np.eye(52),Y[:52]); print('rbf',g,np.mean(abs(pr-Y[52:]),axis=0)[[0,1,4,8]].round(4),end='; ')
 print()
