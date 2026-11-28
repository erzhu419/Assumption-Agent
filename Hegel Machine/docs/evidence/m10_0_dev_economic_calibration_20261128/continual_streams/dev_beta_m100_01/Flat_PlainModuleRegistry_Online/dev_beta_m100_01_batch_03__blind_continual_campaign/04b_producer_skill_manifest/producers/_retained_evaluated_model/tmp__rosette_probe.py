import pickle, numpy as np, itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette'];
print('telemetry keys',list(s.get('aux_manifold_telemetry',{}))); print('single bank first',np.c_[*c['single_bank_sweeps'].values()][:15]); print('design slice nonzero counts',np.count_nonzero(x,axis=1))
def design(x,d,kind):
 terms=[t for k in range(d+1) for t in itertools.combinations_with_replacement(range(4),k) if kind=='full' or len(set(t))<2 or k==2]; return np.column_stack([np.prod(x[:,t],axis=1) for t in terms]), terms
for d,kind in [(1,'full'),(2,'full'),(3,'pure'),(3,'full'),(4,'full'),(5,'pure')]:
 A,t=design(x,d,kind); keep=np.max(np.abs(y),axis=1)<5
 pred=np.zeros_like(y)
 for fold in np.array_split(np.arange(len(x)),5):
  tr=keep.copy(); tr[fold]=False
  pred[fold]=A[fold]@np.linalg.lstsq(A[tr],y[tr],rcond=None)[0]
 r=np.max(abs(pred-y),axis=1); b=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0]; rr=np.max(abs(A@b-y),axis=1)
 print(d,kind,'clean n',sum(keep),'CV median/max',np.median(r[keep]),max(r[keep]),'fold maxima',[round(max(r[z][keep[z]],default=0),6) for z in np.array_split(np.arange(len(x)),5)],'train med/max',np.median(rr[keep]),max(rr[keep]));
 if d==3 and kind=='full':
  for j,col in enumerate(b.T): print('coef',j,[(k,round(v,6)) for k,v in zip(t,col) if abs(v)>0.002]); print('largest train residual rows',sorted(enumerate(rr),key=lambda a:-a[1])[:12]);
