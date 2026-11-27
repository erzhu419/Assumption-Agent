import pickle,numpy as np,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']
def design(x,terms):
 return np.column_stack([np.prod(x[:,t],axis=1) for t in terms])
terms=[t for d in range(4) for t in itertools.combinations_with_replacement(range(4),d)]
a=design(x,terms[:5]); f=np.linalg.lstsq(a,y,rcond=None)[0]; r=np.linalg.norm(y-a@f,axis=1); good=r<6*np.median(r); print('excluded:',np.flatnonzero(~good));
for name,ts in [('affine',terms[:5]),('separable cubic',[t for t in terms if len(set(t))<=1]),('affine+pure cubic',[t for t in terms if len(t)<2 or (len(t)==3 and len(set(t))==1)]),('full quadratic',terms[:15]),('full cubic',terms)]:
 a=design(x[good],ts); f=np.linalg.lstsq(a,y[good],rcond=None)[0]; r=y[good]-a@f; h=(np.linalg.qr(a)[0]**2).sum(1); loo=r/(1-h[:,None]);print(name,'rmse/max',np.sqrt(np.mean(loo**2)),abs(loo).max(),'train',abs(r).max());
 if name=='full cubic':
  for t,row in zip(ts,f): print(t,np.round(row,7))
print('pure rosette rows'); print(np.c_[x,y][np.count_nonzero(x,axis=1)<=1].round(5));
