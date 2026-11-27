import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 exec(open('/tmp/m99_probe.py').read())
p,h,D=cc; Y=np.column_stack((ch['strain_curves_obs']-ch['strain_0'][:,None],ch['instant_rate_obs']))
np.set_printoptions(precision=5,suppress=True,linewidth=200)
labs=ch['lab_id']; r=D@theta[2:]; F=np.column_stack((-np.expm1(-t[:,None]/tau),t)); F=np.vstack((F,np.r_[1/tau,1]));
print('intrinsic',tau,theta); print('fixed/shaped closure comparison');
for name,B in [('intrinsic',F),('free',np.vstack((np.column_stack((-np.expm1(-t[:,None]/np.array([.5025,2.5386])),t)),[1/.5025,1/2.5386,1]))),('intrinsic plus quadratic',np.column_stack((F,np.r_[t*t,0]))),('intrinsic plus third exp',np.column_stack((F,np.r_[-np.expm1(-t/1.2),1/1.2]))),('intrinsic plus attenuated drift',np.column_stack((F,np.r_[t/(1+.3*t),1])) )]:
 E=Y-(np.linalg.lstsq(B,Y.T,rcond=None)[0].T@B.T); print(name,np.sqrt(np.mean(E**2)),np.max(abs(E)),np.sqrt(np.mean(E**2,axis=0)))
base=np.column_stack((h,r,np.ones(len(h)))); co=np.linalg.lstsq(base,Y,rcond=None)[0]; print('time, hold kernel, feature drift kernel, intercept, residual RMS'); print(np.column_stack((np.r_[t,0],co.T,np.sqrt(np.mean((Y-base@co)**2,axis=0))))); print('hold minus both zero-memory modes; drift minus linear',np.column_stack((co[0]-F[:,:2]@theta[:2],co[1]-F[:,2]))); print('empirical joint basis sv',np.linalg.svd(Y,compute_uv=False))

def cv(M):
 E=np.zeros_like(Y)
 for l in np.unique(labs):
  keep=labs!=l; E[~keep]=Y[~keep]-M[~keep]@np.linalg.lstsq(M[keep],Y[keep],rcond=None)[0]
 return np.sqrt(np.mean(E[:,:-1]**2)),np.max(abs(E[:,:-1])),np.max(abs(E[:,-1]))
for name,M in [('hold, drift',base),('hold, drift, past',np.column_stack((base,p))),('hold, features',np.column_stack((h,D))),('hold drift product',np.column_stack((base,h*r))),('drift rationals',np.column_stack((base,r*r,r*r*r,h*r))),('hold nonlinear',np.column_stack((base,h*h,np.tanh(h))))]: print(name,cv(M))
import json
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); a,b=[np.asarray(l['facility_telemetry_trace']) for l in log]; print('apparatus start/current means and stds',a.mean(0),a.std(0),b.mean(0),b.std(0)); print('apparatus first two',a[:2],b[:2]); print('feature ranges',np.quantile(r,[0,.5,1]));
