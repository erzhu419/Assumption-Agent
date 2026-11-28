import pickle,itertools
import numpy as np
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=200)
x=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=x['calibration_load_cells']; t=c['true_channel_loads']; r=c['bridge_raw']; s=c['single_channel_sweeps']; clean=np.max(abs(r),axis=1)<5
fams={'tanh':lambda u,a,b: np.tanh(a*u),'atan':lambda u,a,b:np.arctan(a*u),'tanh+cubic':lambda u,a,b: np.tanh(a*u)+b*u**3,'atan+cubic':lambda u,a,b:np.arctan(a*u)+b*u**3,'tanh+sin':lambda u,a,b:np.tanh(a*u)+b*np.sin(3*u),'tanh+sinfit':lambda u,a,b:np.tanh(a*u)+b[0]*np.sin(b[1]*u),'rational+cubic':lambda u,a,b:u/(1+a*u*u)+b*u**3}
for name,f in fams.items():
 print(name)
 for k in range(4):
  m=(s['lab_id']==k)&(abs(s['cmd_stress'])<5); u=s['true_stress'][m]; y=s['cmd_stress'][m]; ma=clean & (t[:,k]!=0) & (np.count_nonzero(t,axis=1)==1); u=np.r_[u,t[ma,k]]; y=np.r_[y,r[ma,k]]
  def pred(p,u):return p[0]+p[1]*u+p[2]*f(u,p[3],p[4:] if name.endswith('sinfit') else p[4])
  p=least_squares(lambda p:pred(p,u)-y, [0,.3,.5,1.8,0]+([3.] if name.endswith('sinfit') else []),max_nfev=3000).x
  print(k,p,'rmse',np.mean((pred(p,u)-y)**2)**.5)
print('mixed count and spike index',sum(clean),np.where(~clean)[0]); print('all near-zero rows',np.c_[t,r][np.linalg.norm(t,axis=1)<.4]); print('trial shape in archive source:')
