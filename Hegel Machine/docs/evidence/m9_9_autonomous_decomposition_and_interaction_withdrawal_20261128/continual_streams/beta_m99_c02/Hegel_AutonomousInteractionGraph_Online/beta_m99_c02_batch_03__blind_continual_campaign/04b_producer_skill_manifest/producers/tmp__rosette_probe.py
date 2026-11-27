import itertools,json,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; b=s['sensor_calibration_sweeps']; x=b['ref_rosette']; y=b['raw_rosette']; n=len(x)
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
if isinstance(a,dict): a=next(v for v in a.values() if isinstance(v,list))
for e in a:
 t=np.array(e['facility_telemetry_trace']); print(e['completed_campaign_short_name'],'mean',np.round(t.mean(0),3),'last',np.round(t[-1],3))
print('bad rows (gross raw)',np.flatnonzero(np.max(abs(y),axis=1)>5)); print('axis sweeps',np.round(np.c_[x,y][:16],6))
def basis(x,p): return np.prod(x[:,None,:]**p[None,:,:],axis=2)
for name,deg,odd in [('affine',1,False),('quadratic',2,False),('odd cubic',3,True),('full cubic',3,False),('odd quintic',5,True)]:
 p=np.array([i for i in itertools.product(range(deg+1),repeat=4) if sum(i)<=deg and (not odd or sum(i)%2 or not sum(i))]); z=basis(x,p); clean=np.max(abs(y),axis=1)<5
 if len(p)>=clean.sum()-15: continue
 errors=np.full((n,4),np.nan)
 for f in range(5):
  tr=clean&(np.arange(n)%5!=f); te=clean&~tr
  co=np.linalg.lstsq(z[tr],y[tr],rcond=None)[0]; errors[te]=z[te]@co-y[te]
 c=np.linalg.lstsq(z[clean],y[clean],rcond=None)[0]
 print(name,'OOF rms',np.sqrt(np.nanmean(errors**2,axis=0)),'max',np.nanmax(abs(errors),axis=0))
 for j in range(4): print(j,[(tuple(p[i]),round(c[i,j],6)) for i in np.argsort(-abs(c[:,j]))[:10]])
