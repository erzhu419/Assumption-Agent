import pickle,numpy as np
from scipy.special import erf
from scipy.stats import median_abs_deviation as mad
for p in pickle.load(open('/holdout_task/public_surveys.pkl','rb')):
 s=p['survey']; w=s['W_m']; r=s['R_plume_m']; z=s['z_src_m'][:,None]; d=s['z_rcv_m'][None,:]-z; l=np.hypot(w,d); x,h=p['gt']['plume_xz_m']; u=(w*x+d*(h-z))/l; v=(-d*x+w*(h-z))/l
 k=r*np.sqrt(np.pi/2)*np.exp(-0.5*(v/r)**2)*(erf((l-u)/(np.sqrt(2)*r))+erf(u/(np.sqrt(2)*r)))
 a=np.stack([l,-d*d/l,k],axis=-1).reshape(-1,3)
 y=(s['T_obs_s']-p['gt']['rcv_clock_bias_s']).ravel()*1000
 b=np.linalg.lstsq(a,y,rcond=None)[0]
 for _ in range(12):
  e=y-a@b; weights=1/np.hypot(1,e/.25); b=np.linalg.lstsq(a*weights[:,None],y*weights,rcond=None)[0]
 e=y-a@b; ok=abs(e)<3
 print('coefficients (ms/m)',b,'MAD (ms)',mad(e[ok],scale='normal'),'outliers (ms)',np.round(np.sort(e[~ok]),4),'inlier range',np.round([e[ok].min(),e[ok].max()],4))
 print('clock sum',sum(p['gt']['rcv_clock_bias_s']))
