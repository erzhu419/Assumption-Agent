import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; rng=np.random.default_rng(17)
for key in ['panel_a','panel_b']:
 u,v=map(s[key].get,('u','v')); n,d=u.shape; ix=list(itertools.combinations(range(d),2)); D=np.ones((n,1)); cross=np.array([u[:,i]*u[:,j] for i,j in ix]).T
 bases={'aff':(u,), 'square':(u*u,), 'abs':(abs(u),), 'aff+sq':(u,u*u),'aff+abs':(u,abs(u)), 'aff+cubic':(u,u**3),'addpoly3':(u,u**2,u**3),'fullquad':(u,u**2,cross),'fullquad+cubic':(u,u**2,cross,u**3),'aff+tanh':(u,np.tanh(u)), 'aff+sin':(u,np.sin(u)), 'aff+cos':(u,np.cos(u)), 'aff+foldrat':(u,u*u/(1+u*u)), 'aff+abs+sq':(u,abs(u),u*u),'aff+cubic+mix':(u,u**3,cross)}
 print('\n',key, 'row norms v', np.round(np.linalg.norm(v,axis=1),2)); print('affine inliers first pass')
 for name,b in bases.items():
  X=np.column_stack((D,*b)); k=X.shape[1]; t=max(k+1,int(n*.75)); best=None
  for _ in range(240):
   keep=rng.choice(n,k+1,replace=False)
   for _ in range(4):
    cf=np.linalg.lstsq(X[keep],v[keep],rcond=None)[0]; e=np.linalg.norm(X@cf-v,axis=1); keep=np.argsort(e)[:t]
   err=np.mean(e[keep]);
   if best is None or err<best[0]:best=err,e,cf,keep
  _,e,c,I=best; loo=e[I]/np.maximum(1-np.sum(X[I]*np.linalg.pinv(X[I]).T,axis=1),1e-9)
  print(name,'loo75',round(np.median(loo),6), 'all',np.round(np.quantile(e,[.25,.5,.75,.85,.9,1]),5))
  if name in ['fullquad+cubic','aff+cubic','fullquad','aff+sq','aff+abs','aff']:
   print('  coef',np.round(c,3).T.tolist())
