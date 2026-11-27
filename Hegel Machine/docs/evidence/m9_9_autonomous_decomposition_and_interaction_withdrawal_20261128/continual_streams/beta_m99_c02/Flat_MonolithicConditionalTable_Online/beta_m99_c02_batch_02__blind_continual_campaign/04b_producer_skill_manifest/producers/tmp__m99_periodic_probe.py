exec(open('/tmp/m99_wall_probe.py').read().split("for mode in ('paired','wrapped'):")[0])
p=np.zeros(12)
for j,(A,k,v,h,r) in enumerate(base):
 face=(k[0]+k[-1])/2
 for b,(edge,near) in enumerate(((0,1),(-1,-2))):
  towards_near=max((-1 if b==0 else 1)*v[edge],0)
  towards_wrap=max((1 if b==0 else -1)*v[edge],0)
  t=2*j+b
  p[t]=((k[edge]+k[near])/2+h*towards_near)/k[edge]
  p[8+t]=(face+h*towards_wrap)/k[edge]
print('periodic coefficients',p.round(8))
e=prediction(p,True)-y
print('unprojected shift x pulse RMS',np.sqrt(np.mean(e.reshape(3,32,3)**2,axis=1)))
print('unprojected pulse max abs',np.max(abs(e),axis=0))
print('pulse projected RMS',np.sqrt(np.mean(err(p,True)**2,axis=0)))
ref,raw=[s['sensor_calibration_sweeps'][key] for key in ('ref_rosette','raw_rosette')]; c=np.c_[np.ones(len(ref)),ref]; coef=np.linalg.lstsq(c,raw,rcond=None)[0]; print('calibration coefficient matrix',coef.round(7)); held=[]
for j in range(5):
 keep=np.arange(len(c))%5!=j; b=np.linalg.lstsq(c[keep],raw[keep],rcond=None)[0]; yp=np.column_stack([np.linalg.solve(b[1:].T,(d['y_pulse_raw'].reshape(-1,4)-b[0]).T).T.ravel() for d in ds]); held.append(np.sqrt(np.mean((yp-prediction(p,True))**2,axis=0)))
print('worst calibration-fold pulse RMS',np.max(held,axis=0))
