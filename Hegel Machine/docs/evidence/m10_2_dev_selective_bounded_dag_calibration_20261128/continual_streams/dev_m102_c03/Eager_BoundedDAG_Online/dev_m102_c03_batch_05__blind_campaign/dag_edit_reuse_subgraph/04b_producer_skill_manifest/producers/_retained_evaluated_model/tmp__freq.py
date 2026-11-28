exec(open('/tmp/readout_probe.py').read().split("for kind in ['diag'")[0]);
from scipy.optimize import minimize
for k in ['cyc','aff','diag','raw']:
 z=x if k=='raw' else cal(x,k); p=pp(z); h0=np.c_[np.ones(len(z)),p];
 for mode in [0,1]:
  def f(a):
   if mode==0:return np.c_[h0,np.cos(z*a)]
   return np.c_[h0,np.cos(z*a[:4]+a[4:])]
  for j in [0,1,2]:
   fit=np.arange(52); y=Y[:,j];
   def err(a,ret=False):
    H=f(a); b=np.linalg.lstsq(H[fit],y[fit],rcond=None)[0]; e=H@b-y;return (np.abs(e[52:]).mean(), b) if ret else np.mean(e[fit]**2)
   opt=min((minimize(err,np.r_[np.repeat(freq,4),np.zeros(mode*4)],method='L-BFGS-B',bounds=[(.15,8)]*4+[(-3.14,3.14)]*(mode*4)) for freq in [1.,2.4,3.5,5.,6.5]),key=lambda o:o.fun); print(k,mode,j,np.round(opt.x,3),round(opt.fun**.5,4),np.round(err(opt.x,True)[0],4),np.round(err(opt.x,True)[1],3))
