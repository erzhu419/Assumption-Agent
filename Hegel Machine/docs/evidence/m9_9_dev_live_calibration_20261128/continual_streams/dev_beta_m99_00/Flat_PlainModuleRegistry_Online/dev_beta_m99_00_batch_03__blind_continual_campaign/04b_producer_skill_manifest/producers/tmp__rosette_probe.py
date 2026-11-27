import pickle,itertools,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; xb=np.eye(4)[b['bank_id']]*b['ref_signal'][:,None]
def fit(a,y):
 w=np.linalg.lstsq(a,y,rcond=None)[0]
 for _ in range(20):
  e=np.linalg.norm(y-a@w,axis=1); q=np.minimum(1,(max(np.median(e),1e-12)*1.8/np.maximum(e,1e-15))**2)**.5
  w=np.linalg.lstsq(a*q[:,None],y*q[:,None],rcond=None)[0]
 return w
def poly(x,maxdeg):
 return np.column_stack([np.ones(len(x))]+[x[:,idx].prod(axis=1) for d in range(1,maxdeg+1) for idx in itertools.combinations_with_replacement(range(4),d)])
models={'linear':lambda x:poly(x,1),'quadratic':lambda x:poly(x,2),'diag23':lambda x:np.c_[poly(x,1),x*x,x**3],'quad+cubics':lambda x:np.c_[poly(x,2),x**3],'full3':lambda x:poly(x,3),'quad+tanh':lambda x:np.c_[poly(x,2),np.tanh(x)],'quad+rational':lambda x:np.c_[poly(x,2),x/(1+x*x)],'quad+sin':lambda x:np.c_[poly(x,2),np.sin(x)],'quad+cubics+5':lambda x:np.c_[poly(x,2),x**3,x**5]}
np.set_printoptions(precision=5,suppress=True)
for name,f in models.items():
 a=f(x); w=fit(a,y); e=np.linalg.norm(a@w-y,axis=1); pred=np.empty_like(y)
 for te in np.array_split(np.random.default_rng(12).permutation(len(y)),5):
  tr=np.setdiff1d(np.arange(len(y)),te); pred[te]=a[te]@fit(a[tr],y[tr]);
 cv=np.linalg.norm(pred-y,axis=1); eb=(f(xb)@w)[np.arange(len(xb)),b['bank_id']]-b['raw_readout']; print(name, 'train',np.percentile(e,[25,50,90,97,100]), 'cv',np.percentile(cv,[25,50,90,97,100]),'bank',np.percentile(abs(eb),[50,90,100]));
 if name in ('quad+cubics','quad+tanh','full3'): print('coeffs',w.round(5)); print('top rows',np.argsort(e)[-8:])
print('axes ref/raw',np.c_[x[:16],y[:16]]); print('bank raw',b['raw_readout'].reshape(4,-1))
