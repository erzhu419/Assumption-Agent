exec(open('/tmp/probe.py').read().split('u,v=')[0])
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
for p,k in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][k]; m=np.ones(len(u),bool) if p!='panel_a' else np.max(abs(v),axis=1)<3; u,v=u[m],v[m]; print(p,k,len(u))
 for d in (1,2,3):
  poly=PolynomialFeatures(d); h=poly.fit_transform(u); b=ls(h,v); print('poly',d,'std',np.std(h@b-v),'max',np.max(abs(h@b-v)))
  if d==2: print([(n,np.round(w,3)) for n,w in zip(poly.get_feature_names_out(),b)])
 for name,f in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('sinh',np.sinh)]:
  for j in range(4):
   w=ls(aff(u),v[:,j]); sol=least_squares(lambda b:b[-2]+b[-1]*f(aff(u)@b[:5])-v[:,j],np.r_[w,0,1],max_nfev=400); print(name,j,round(np.std(sol.fun),6),np.round(sol.x,3))
 if p=='panel_a':
  for name,h in [('diag',aff(u)),('sq',np.c_[aff(u),u**2]),('cubic',np.c_[aff(u),u**2,u**3]),('rollproduct',np.c_[aff(u),u*np.roll(u,1,axis=1)]),('tanh',np.c_[aff(u),np.tanh(u)]),('sin',np.c_[aff(u),np.sin(u)])]:
   b=ls(h,v); print(name,np.std(h@b-v),np.round(b,4))
