import pickle,json,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; np.set_printoptions(precision=3,suppress=True,linewidth=180)
z=(abs(y).max(1)<4); print('large sweep rows'); print(np.c_[np.arange(len(x)),x,y][~z]); print('position ranges',[(i, x[i:i+20].min(),x[i:i+20].max(),z[i:i+20].sum()) for i in range(0,len(x),20)])
one=np.ones((len(x),1)); pair=np.stack([x[:,i]*x[:,j] for i in range(4) for j in range(i+1,4)],1); cube=np.stack([x[:,i]*x[:,j]*x[:,k] for i in range(4) for j in range(i,4) for k in range(j,4)],1); designs={'affine':np.c_[one,x],'diagquad':np.c_[one,x,x*x],'diagcube':np.c_[one,x,x*x,x**3],'quad':np.c_[one,x,x*x,pair],'quad+diagcube':np.c_[one,x,x*x,pair,x**3],'cubic':np.c_[one,x,x*x,pair,cube]}
for name,A in designs.items():
 p=np.linalg.lstsq(A[z],y[z],rcond=None)[0]; e=y-A@p; T=A[z]; h=np.einsum('ij,ji->i',T,np.linalg.pinv(T)); loo=e[z]/(1-h[:,None]); print(name,'LOO RMS',np.sqrt(np.mean(loo**2,0)),'worst',abs(loo).max(0),'worst clean rows',np.flatnonzero(z)[np.argsort(np.linalg.norm(loo,axis=1))[-5:]])
 if name in ('diagcube','quad+diagcube'): print('coefs',p)
print('aux'); print(s['aux_manifold_telemetry']); r=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('apparatus half means');
for a in r:
 t=np.asarray(a['facility_telemetry_trace']); print(a['completed_short_name'],np.array([q.mean(0) for q in np.split(t,4)]))
