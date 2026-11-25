import numpy as np,pickle
from scipy.special import erf
from scipy.optimize import least_squares
P=pickle.load(open('/holdout_task/public_surveys.pkl','rb'))
for p in P:
 s=p['survey']; gt=p['gt']; W=s['W_m']; R=s['R_plume_m']; a=s['z_src_m'][:,None]; b=s['z_rcv_m'][None,:]; dz=b-a; L=np.hypot(W,dz); sn=dz/L; cs=W/L; x,z=gt['plume_xz_m']; u=x*cs+(z-a)*sn; v=(z-a)*cs-x*sn
 g=R*np.sqrt(np.pi/2)*np.exp(-v*v/(2*R*R))*(erf((L-u)/(np.sqrt(2)*R))+erf(u/(np.sqrt(2)*R)))
 X=np.stack([L,-L*sn**2,g],axis=-1).reshape(-1,3)
 y=(s['T_obs_s']-gt['rcv_clock_bias_s']).ravel()*1000
 fit=least_squares(lambda q:X@q-y,[.42,.02,.2],loss='cauchy',f_scale=.25)
 r=y-X@fit.x
 print('params ms/m',fit.x,'epsilon',fit.x[1]/fit.x[0]); print('residual quantiles ms',np.round(np.quantile(r,[0,.01,.025,.05,.1,.25,.5,.75,.9,.95,.975,.99,1]),4)); ind=abs(r)>2; print('large residuals',np.round(np.sort(r[ind]),4),'corrected std',np.std(r-np.sign(r)*ind*6)); print('clock sum',sum(gt['rcv_clock_bias_s']))
