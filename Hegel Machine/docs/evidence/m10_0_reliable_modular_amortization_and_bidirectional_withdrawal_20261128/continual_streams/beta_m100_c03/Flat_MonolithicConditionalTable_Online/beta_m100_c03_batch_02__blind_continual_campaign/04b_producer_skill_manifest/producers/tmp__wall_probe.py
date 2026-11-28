import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('archive','/tmp/hegel_continual_archive/beta_m100_c03_batch_01_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=i['study']; d=i['campaign_diagnostics']; w=np.array([p['w_pulse'] for p in d]).T; y=np.column_stack([m._calibrate(s,p['y_pulse_raw']) for p in d]); print('pulse cal range',y.min(),y.max()); c=s['sensor_calibration_sweeps']; ref=c['ref_rosette']; raw=c['raw_rosette']; X=np.c_[np.ones(len(ref)),ref]; B=np.linalg.lstsq(X,raw,rcond=None)[0]; print('cal coefficients',B,'rms',np.sqrt(np.mean((X@B-raw)**2)))
def resp(p):
 axes=[]
 for j,z in enumerate('xy'):
  k=s['kappa_'+z]; h=s['L'+z]/(len(k)-1); rob=s['robin_'+z]; a=m._axis(k,s['vel_'+z],h,0.).tolil(); mult,leak=p[4*j:4*j+2],p[4*j+2:4*j+4];
  for b,n,u,v in zip((0,len(k)-1),(1,len(k)-2),mult,leak):
   a[b,n]=-u*k[b]/h**2; a[b,b]=-a[b,n]+v*rob/h
  axes.append(a.tocsc())
 return m._responses(s,axes)[0]@w
rms=lambda a: np.sqrt(np.mean(a*a,axis=0))
for name,p in [('open',[1,1,1,1]*2),('sealed',[1,1,0,0]*2),('doubled',[2,2,1,1]*2),('double sealed',[2,2,0,0]*2)]:
 print(name,rms(resp(p)-y),'max',abs(resp(p)-y).max())
for name,expand,p in [('leak',lambda q:[1,1,*q[:2],1,1,*q[2:]],[.6]*4),('symmetric',lambda q:[q[0],q[0],q[1],q[1],q[2],q[2],q[3],q[3]],[1,.6,1,.6]),('general',lambda q:q,[1,1,.6,.6]*2)]:
 for fold in [None,0,1,2]:
  train=[k for k in range(3) if k!=fold]; fit=least_squares(lambda q:(resp(expand(q))[:,train]-y[:,train]).ravel(),p,gtol=1e-13); print(name,fold,fit.x,'rms',rms(resp(expand(fit.x))-y),'sv_min',np.linalg.svd(fit.jac,compute_uv=False)[-1])
