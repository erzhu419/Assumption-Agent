import numpy as np,pickle
from scipy.optimize import least_squares, minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import logit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; n=len(x)
p=s['panel_a']; u=p['u']; v=p['v']; good=np.linalg.norm(v-u,axis=1)<1; off,scale=np.array([np.linalg.lstsq(np.c_[np.ones(sum(good)),u[good,j]],v[good,j],rcond=None)[0] for j in range(4)]).T
print('calibration',off,scale,sum(good)); print('times',t)
_,_,M=np.linalg.svd(z,full_matrices=False); Z=z@M[:2].T; Y=np.c_[y,Z]; 
for j in (1,2):
 def basis(p):
  return np.stack([-np.expm1(-p[0]*t), t if j==1 else -np.expm1(-p[1]*t)],axis=1)
 def res(p):
  A=basis(p); return (z-(z@np.linalg.pinv(A.T))@A.T).ravel()
 for start in ([1.2,0.25],[2.5,0.5],[1,1.8]):
  fit=least_squares(res,start,bounds=(0.00001,20)); print('temporal',j,fit.x,np.std(fit.fun))
for nm,X in [('raw',x),('calibrated',x*scale+off),('inverse',(x-off)/scale)]:
 F=PolynomialFeatures(3); U=F.fit_transform(X); print(nm)
 for k in (52,72):
  a=np.linalg.lstsq(U[:k],Y[:k],rcond=None)[0]; print('train n',k,'rms',np.std(U[:k]@a-Y[:k],axis=0),'cv MAE',np.mean(abs(U[52:]@a-Y[52:]),axis=0))
  if k==72:
   for c in range(3):
    ix=np.argsort(abs(a[:,c]))[::-1]; print(c,[(str(F.get_feature_names_out()[i]),round(a[i,c],4)) for i in ix[:22]])
for pkey in ['panel_b','panel_c']:
 p=s[pkey]; u=p['u']; F=PolynomialFeatures(2).fit_transform(u); L=np.c_[np.ones(len(u)),u];
 for key in p:
  if key=='u':continue
  v=p[key]; print(pkey,key,'quadratic coeffs:');print(np.round(np.linalg.lstsq(F,v,rcond=None)[0],4));
  for i in range(4):
   w=v[:,i];
   for label,link,start in [('tanh',np.tanh,1.4),('sin',np.sin,1.8),('softsign',lambda a:a/(1+abs(a)),1.8),('cubic',lambda a:a**3,1.0),('square',lambda a:a*a,1.)]:
    def pred(p):return p[0]+p[1]*link(L@p[2:]);
    a=np.linalg.lstsq(L,w,rcond=None)[0]; opts=[least_squares(lambda p:pred(p)-w,np.r_[0,start,f*a],max_nfev=400) for f in (0.5,1.5,3)]; print(i,label,round(min(np.std(q.fun) for q in opts),5),end='; ')
   print()
