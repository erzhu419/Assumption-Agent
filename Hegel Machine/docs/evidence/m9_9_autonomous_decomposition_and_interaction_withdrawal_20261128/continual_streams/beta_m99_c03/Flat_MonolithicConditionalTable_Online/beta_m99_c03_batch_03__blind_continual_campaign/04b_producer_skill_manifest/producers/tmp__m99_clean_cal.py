import itertools, json, pickle
import numpy as np

s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; b=c['single_bank_sweeps']; n=len(x)
xb=np.zeros((len(b['bank_id']),4)); xb[np.arange(len(xb)),b['bank_id']]=b['ref_signal']
yb=np.full_like(xb,np.nan); yb[np.arange(len(xb)),b['bank_id']]=b['raw_readout']; xx=np.vstack((x,xb)); yy=np.vstack((y,yb)); good=np.isfinite(yy)&(abs(yy)<8)
fold=np.r_[np.arange(n)%5,np.arange(len(xb))%5]; dense=np.count_nonzero(xx,axis=1)>1
for deg in range(1,6):
 for kind in (['all'] if deg!=3 else ['all','own-cube','radial']):
  terms=[t for d in range(deg+1) for t in itertools.combinations_with_replacement(range(4),d) if kind!='own-cube' or len(t)<3 or len(set(t))==1]; A=np.column_stack([np.prod(xx[:,t],axis=1) for t in terms]); E=np.full_like(yy,np.nan); fits=[]
  for k in range(4):
   D=A if kind!='radial' else np.c_[np.ones(len(xx)),xx,xx[:,k]**3,xx[:,k]*(np.sum(xx**2,axis=1)-xx[:,k]**2),*[xx[:,i]*xx[:,j] for i,j in itertools.combinations(range(4),2)]]
   for f in range(6):
    m=good[:,k]&(fold!=f); q=np.linalg.lstsq(D[m],yy[m,k],rcond=None)[0]; h=good[:,k]&(fold==f); E[h,k]=D[h]@q-yy[h,k]
   fits.append(q)
  print(deg,kind,'dense RMSE',np.round(np.sqrt(np.nanmean(E[dense]**2,axis=0)),6),'max',np.round(np.nanmax(abs(E),axis=0),6))
  if deg==3 and kind=='all':
   for k,q in enumerate(fits): print('channel',k,[(t,round(float(v),7)) for t,v in zip(terms,q) if abs(v)>1e-7])
a=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('log tail:')
for e in a[-4:]: print({k:v for k,v in e.items() if k!='facility_telemetry_trace'})
