import numpy as np,pickle,sys
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_a03_batch_04_solver as a
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; af=a._affine; fit=a._fit
np.set_printoptions(precision=4,suppress=True)
cal=a._calibration(s['panel_a']); print('cal',cal); z=a._decode(x,cal); print('decoded range',z.min(0),z.max(0),'raw range',x.min(0),x.max(0)); print('traj sv',np.linalg.svd(T,compute_uv=False)); print('summary',tr['summary'])
def cv(q,tag):
 q=np.asarray(q); p=q[52:]@fit(q[:52],np.c_[y,T][:52]); print(tag,q.shape[1],np.mean(abs(p[:,0]-y[52:])),np.mean(abs(p[:,1:]-T[52:])))
cv(af(x),'raw'); cv(af(z),'inv'); cv(af(np.tanh(z)),'tanh'); cv(af(z/np.sqrt(1+z*z)),'soft');
for panel in ('panel_b','panel_c'):
 u=s[panel]['u'];
 for key,v in s[panel].items():
  if key=='u': continue
  print(panel,key)
  for name,f in [('linear',lambda x:x),('tanh',np.tanh),('sin',np.sin),('cubic',lambda x:np.c_[x,x**3]),('quad',lambda x:np.c_[x,x**2]),('dead',lambda x:np.sign(x)*np.maximum(abs(x)-.22,0)**1.25)]:
   q=af(f(u)); c=fit(q,v); w=af(f(z))@c; print(name,'panel rmse',np.sqrt(np.mean((q@c-v)**2))); cv(af(w),name); cv(np.c_[af(w),w[:,0]*w[:,2]+w[:,1]*w[:,3]],name+' readout')
