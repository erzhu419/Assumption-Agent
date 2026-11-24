import numpy as np,pickle
from scipy.ndimage import correlate1d
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
cases=pickle.load(open('/holdout_task/public_tat.pkl','rb'))
for i,e in enumerate(cases):
 s=e['scan']; g=e['gt']; y=s['signals']; xy=s['sensor_xy_mm']; pin=s['calib_pin_xy_mm']; scale=s['fs_hz']/1000; j=np.arange(len(xy)); dp=np.linalg.norm(xy-pin,axis=1); t=np.arange(len(y)); u=np.arange(-28,29)/4.4; k=-u*np.exp(-u*u/2); n=k@k
 def matched(y): return CubicSpline(t,correlate1d(y,k,axis=0,mode='constant'))
 m=matched(y)
 def diag(m,ts):
  ix=np.floor(ts).astype(int).clip(0,len(t)-2); dt=ts-ix; a=m.c[:,ix,j]; return ((a[0]*dt+a[1])*dt+a[2])*dt+a[3]
 def pscore(c,keep):
  z=diag(m,dp*scale/c)[keep]; return -np.sum(z*z)
 def getpin(c,keep):
  cs=np.arange(1435,1546,1); b=cs[np.argmin([pscore(v,keep) for v in cs])]; return minimize_scalar(pscore,args=(keep,),bounds=(max(b-1,1435),min(b+1,1545)),method='bounded').x
 a=np.arange(-32,32.1,0.75); xg,yg=np.meshgrid(a,a); q=np.c_[xg.ravel(),yg.ravel()]; q=q[(q*q).sum(1)<32**2]; dist=np.linalg.norm(q[:,None]-xy,axis=2)
 def focus(c,keep):
  pm=dp*scale/c; amp=diag(m,pm)/n; v=(t[:,None]-pm)/4.4; z=y+v*np.exp(-v*v/2)*amp
  out=[]
  for _ in range(2):
   tm=dist*scale/c; b=diag(matched(z),tm); sc=np.maximum(b*np.sign(amp),0)[:,keep].sum(1); idx=sc.argmax(); cen=tm[idx]; a=diag(matched(z),cen)/n; v=(t[:,None]-cen)/4.4; z=z+v*np.exp(-v*v/2)*a; out.append(q[idx])
  err=np.linalg.norm(np.array(out)[:,None]-np.array(g['absorbers_xy_mm']),axis=2); return max(err.min(0).max(),err.min(1).max())
 full=np.ones(len(xy),bool); cs=getpin(None,full); jack=[getpin(None,j!=v) for v in j]; print(i,'pin speed',cs,'worst omitted pin speed',max(abs(np.array(jack)-g['sound_speed_m_per_s'])),'grid loc calibrated vs nominal',focus(cs,full),focus(1480,full),'max grid LOOCV',max(focus(c,j!=v) for v,c in enumerate(jack)))
