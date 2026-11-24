import pickle,numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares
R=pickle.load(open('/holdout_task/public_tat.pkl','rb'))
for rec in R:
 s=rec['scan']; y=s['signals']; sens=s['sensor_xy_mm']; pin=np.array(s['calib_pin_xy_mm']); fs=s['fs_hz']; cand=[]
 for j in range(16):
  yy=y[:,j]; pk,_=find_peaks(abs(yy),distance=7,prominence=.45); vals=[]
  for k in pk:
   if 100<k<1150 and k+7<len(yy) and max(abs(yy[k+7:k+12]))>abs(yy[k])*.45:
    t=k+4.5
    if not vals or t-vals[-1]>8: vals.append(t)
  cand.append(np.array(vals))
 print('candidates',cand)
 def residual(z):
  p=z[:2]; q=z[2:4]; c=z[4]; off=z[5]; out=[]
  for src in (p,q):
   direct=off+np.linalg.norm(sens-src,axis=1)*fs*1e-6/c
   refl=off+(np.linalg.norm(src-pin)+np.linalg.norm(sens-pin,axis=1))*fs*1e-6/c
   for j in range(16):
    if len(cand[j]):
     for pred in (direct[j],refl[j]):
      ds=cand[j]-pred; out.append(ds[np.argmin(abs(ds))])
  return np.array(out)
 best=None
 for c0 in np.linspace(1.435,1.545,4):
  for off in [150,250,350,450]:
   for seed in range(10):
    rng=np.random.default_rng(seed+int(off)); z=np.r_[rng.uniform(-25,25,4),c0,off]
    fit=least_squares(residual,z,bounds=([-31]*4+[1.435,0],[31]*4+[1.545,800]),loss='soft_l1',f_scale=3,max_nfev=100)
    score=np.median(abs(residual(fit.x)))
    if best is None or score<best[0]: best=(score,fit.x)
 print('GT',rec.get('gt'),'best',best)
