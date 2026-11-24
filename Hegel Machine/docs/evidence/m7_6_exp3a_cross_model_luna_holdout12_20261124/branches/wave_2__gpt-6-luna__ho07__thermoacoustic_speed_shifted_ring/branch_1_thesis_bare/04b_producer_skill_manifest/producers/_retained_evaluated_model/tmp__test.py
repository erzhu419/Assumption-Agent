import pickle,numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares
R=pickle.load(open('/holdout_task/public_tat.pkl','rb'))
for r in R:
 s=r['scan']; y=s['signals']; sens=s['sensor_xy_mm']; fs=s['fs_hz']
 t=np.arange(-12,13); temp=-t*np.exp(-.5*(t/3.2)**2); temp-=temp.mean(); temp/=np.linalg.norm(temp)
 cs=[]
 for sig in y.T:
  corr=np.correlate(sig-np.mean(sig),temp,'same'); pk,_=find_peaks(abs(corr),distance=12,prominence=.2)
  cs.append(np.array(sorted(pk[(pk>250)&(pk<800)]),float))
 print('gt',r['gt']); print('candidates',cs)
 def res(z):
  pts=z[:4].reshape(2,2); c=z[4]; off=z[5]; out=[]
  for p in pts:
   pred=off+np.linalg.norm(sens-p,axis=1)*fs*1e-6/c
   for j in range(16):
    if len(cs[j]):
     d=cs[j]-pred[j]; out.append(d[np.argmin(abs(d))])
  return out
 best=None
 for k in range(80):
  z=[*np.random.uniform(-25,25,4),np.random.uniform(1.435,1.545),np.random.uniform(350,450)]
  fit=least_squares(res,z,bounds=([-31]*4+[1.435,200],[31]*4+[1.545,600]),max_nfev=300)
  score=np.median(abs(fit.fun))
  if best is None or score<best[0]: best=(score,fit.x)
 print('fit',best)
