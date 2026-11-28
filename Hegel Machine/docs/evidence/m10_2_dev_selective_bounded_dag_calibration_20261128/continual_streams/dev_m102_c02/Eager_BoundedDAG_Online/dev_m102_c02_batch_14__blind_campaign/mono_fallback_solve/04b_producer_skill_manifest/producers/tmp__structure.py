import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=160)
for ky in ['panel_a','panel_b','panel_c']:
 p=s[ky];u=p['u']; po=PolynomialFeatures(3);P=po.fit_transform(u); terms=po.get_feature_names_out();deg=po.powers_.sum(1); pure=(po.powers_>0).sum(1)==1
 for k,v in p.items():
  if k=='u':continue
  X=P; m=np.ones(len(u),bool)
  if ky=='panel_a':
   for j in range(4):
    b=least_squares(lambda b:X[:,:5]@b-v[:,j],np.linalg.lstsq(X[:,:5],v[:,j],rcond=None)[0],loss='soft_l1',f_scale=.13).x
    m &= abs(X[:,:5]@b-v[:,j])<1
   print('inliers',m.sum(),'excluded',np.where(~m)[0])
  X=X[m];V=v[m]; print(ky,k)
  for name,a in [('linear+purecub',(deg<=1)|((deg==3)&pure)),('odd cubic',(deg<=1)|(deg==3)),('pure powers',(deg==0)|pure),('quadratic',deg<=2),('all cubic',deg<=3)]:
   A=X[:,a];B=np.linalg.lstsq(A,V,rcond=None)[0];e=A@B-V;h=np.sum(A*np.linalg.pinv(A).T,axis=1);print(name,np.round(np.mean(abs(e),0),4),np.round(np.mean(abs(e)/np.maximum(1-h[:,None],1e-8),0),4));
   if ky=='panel_a' and name in ['linear+purecub','odd cubic','all cubic'] or ky=='panel_c' and k=='v2' and name=='quadratic':
    for j in range(4):print(j,[(terms[a][i],round(B[i,j],4)) for i in np.argsort(abs(B[:,j]))[:-10:-1]])
t=s['train']['traj'];ts=s['eval_times'];p=np.linalg.svd(t,full_matrices=False)[2]; print('time mode 0',p[0]/p[0,-1]);print('log',np.log1p(2.15*ts)/np.log1p(2.15*ts[-1]));print('exp',-np.expm1(-1.25*ts)/(-np.expm1(-1.25*ts[-1])));print('time modes',p[:3]);print('correlation y,pc',np.corrcoef(np.c_[s['train']['y'],t@p[:3].T].T));
