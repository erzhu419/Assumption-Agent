import numpy as np,pickle
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=145)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
u,v=[s['panel_a'][k] for k in ['u','v']]; keep=np.linalg.norm(v,axis=1)<5; u,v=u[keep],v[keep]
for j in range(4):
 a,b=u[:,j],u[:,(j+1)%4]; f=np.c_[a,b]; print('A channel',j)
 for deg in [1,2,3,4,5]:
  pp=PolynomialFeatures(deg); F=pp.fit_transform(f); w=ls(F,v[:,j]); print(deg, np.std(F@w-v[:,j]),list(zip(pp.get_feature_names_out(),w.round(5))) if deg<=3 else '')
 print('cyclic rational/sat alternatives')
 for fun in [lambda a:np.tanh(a),lambda a:a*np.exp(-a*a),lambda a:a/(1+a*a),lambda a:np.sin(a),lambda a:a/(1+.5*a*a)]:
  F=np.c_[np.ones(len(a)),a,b,a*b,a**3,fun(a),fun(b),a*fun(b),b*fun(a)]; w=ls(F,v[:,j]); print(np.std(F@w-v[:,j]),w)
for k in ['panel_b','panel_c']:
 u=s[k]['u']; print(k)
 for n,v in s[k].items():
  if n=='u':continue
  for desc,F in [('full quad',PolynomialFeatures(2).fit_transform(u)),('linear + cyclic nonlinear',np.c_[np.ones(len(u)),u,u*np.roll(u,-1,axis=1),u**2,u**3]),('linear + sine',np.c_[np.ones(len(u)),u,np.sin(u),u*np.roll(u,-1,axis=1)])]:
   w=ls(F,v); H=np.sum(F*np.linalg.pinv(F).T,axis=1); print(n,desc,'rms',np.std(F@w-v,axis=0),'loo',np.std((F@w-v)/(1-H[:,None]),axis=0));
   if desc=='full quad':print(w)
