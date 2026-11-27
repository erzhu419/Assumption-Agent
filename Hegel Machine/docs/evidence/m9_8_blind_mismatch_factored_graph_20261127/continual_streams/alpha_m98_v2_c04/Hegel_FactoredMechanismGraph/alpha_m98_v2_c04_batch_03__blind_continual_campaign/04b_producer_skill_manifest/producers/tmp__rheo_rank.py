from rheo_probe import *
from scipy.optimize import least_squares
B=np.c_[yc-(base-(th[2]+xg@th[3:])[:,None]*t), yr-(rbase-(th[2]+xg@th[3:]))]; D=th[2]+xg@th[3:]; labg=g['lab_id']; missing=np.isnan(g['coupon_features_raw']).any(1)
u,s,v=np.linalg.svd(B,full_matrices=False); print('t',t,'dt',dt,'singular values',np.round(s,5)); f=v[0]/v[0,-1]; amps=B@f/(f@f); print('onset-normalized profile',np.round(f,5)); print('amp, input drift, diff, rate residual after rank-1, lab, missing, stratum\n',np.round(np.c_[amps,D,amps-D,B[:,-1]-amps,labg,missing,g['stratum']],4)); print('last three temporal modes\n',np.round(v[:3],4));
for mode in ('sat','scaled_sat','sat_linear','double_sat','tanh','log','rat'):
 def curve(z):
  k=np.exp(z[0]); a=z[1] if len(z)>1 else 1.; F=-np.expm1(-k*t)/k
  if mode=='sat_linear': F=a*F+(1-a)*t
  elif mode=='double_sat': j=np.exp(z[2]); F=a*F+(1-a)*(-np.expm1(-j*t)/j)
  elif mode=='scaled_sat':F=a*F
  elif mode=='tanh': F=np.tanh(k*t)/k
  elif mode=='log': F=np.log1p(k*t)/k
  elif mode=='rat': F=t/(1+k*t)
  return F
 ini={'scaled_sat':[0,1],'sat_linear':[0,1],'double_sat':[0,.8,-1]}.get(mode,[0]); z=least_squares(lambda z:curve(z)-f[:-1],ini).x; print(mode,np.round(z,5),round(np.linalg.norm(curve(z)-f[:-1]),6))
