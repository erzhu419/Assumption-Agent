exec(open('/tmp/periodic_probe.py').read().split("for a in ('x','y'):")[0])
from scipy.optimize import minimize
rhs=src.transpose(0,2,1).reshape(len(src),-1).T
raw=s['y_obs_raw']; obs=cal(raw); Z=s['Z_nuisance']; box=s['u_box']; budget=s['total_budget']; M=len(src)
x=s['sensor_calibration_sweeps']['ref_rosette']; yraw=s['sensor_calibration_sweeps']['raw_rosette']; powers=m._powers(3); c=np.linalg.lstsq(m._features(x,powers),yraw,rcond=None)[0]; print('poly train',np.sqrt(np.mean((m._features(x,powers)@c-yraw)**2,axis=0)))
def mix(v):return (m._features(v.reshape(-1,4),powers)@c).ravel()
for name,wall in [('open',x0),('periodic',np.r_[expected('x'),expected('y')])]:
 A=pred(wall)
 for order in ['calibrated','raw_after','raw_before']:
  def predict(p):
   a=A@p[:M]; z=Z@p[M:]
   if order=='calibrated':return a+z-obs
   if order=='raw_after':return mix(a)+z-raw
   return mix(a+z)-raw
  p0=np.r_[np.full(M,budget/M),np.linalg.lstsq(Z,obs-A@np.full(M,budget/M),rcond=None)[0]]
  fit=minimize(lambda p:np.mean(predict(p)**2)*1e4,p0,method='SLSQP',bounds=[(0,box)]*M+[(None,None)]*Z.shape[1],constraints=[{'type':'eq','fun':lambda p:p[:M].sum()-budget}],options={'ftol':1e-12}); e=predict(fit.x).reshape(3,-1,4); print(name,order,fit.success,'shift',np.sqrt(np.mean(e**2,axis=(1,2))).round(7),'bank',np.sqrt(np.mean(e**2,axis=(0,1))).round(7),'shift means',e.mean(axis=1).round(5),'weights',fit.x[:M].round(4))
