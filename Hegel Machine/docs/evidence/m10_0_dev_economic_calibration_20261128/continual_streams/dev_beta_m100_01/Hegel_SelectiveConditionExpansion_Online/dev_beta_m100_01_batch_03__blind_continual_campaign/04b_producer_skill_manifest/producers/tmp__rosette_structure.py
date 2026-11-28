import json,pickle,numpy as np
from itertools import combinations_with_replacement
np.set_printoptions(precision=5,suppress=True,linewidth=125)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; b=c['single_bank_sweeps']; print('single bank keys',list(b));
def design(x,terms):
 return np.column_stack([np.prod(x[:,t],axis=1) if t else np.ones(len(x)) for t in terms])
def robust(a,y):
 coef=np.linalg.lstsq(a,y,rcond=None)[0]
 for _ in range(35):
  r=np.linalg.norm(a@coef-y,axis=1); w=np.minimum(1.,3*np.median(r)/np.maximum(r,1e-13)); coef=np.linalg.lstsq(a*w[:,None],y*w[:,None],rcond=None)[0]
 return coef
base=[()]+[(i,) for i in range(4)]; models={'linear':base,'diagonal 2+3':base+[(i,)*k for k in (2,3) for i in range(4)],'mixed 2':base+list(combinations_with_replacement(range(4),2)),'mixed 3':base+sum([list(combinations_with_replacement(range(4),k)) for k in (2,3)],[]),'odd mixed 3':base+list(combinations_with_replacement(range(4),3))}
a=design(x,base); ra=np.linalg.norm(a@robust(a,y)-y,axis=1); print('affine row residual sorted',np.sort(ra)); print('calibration row norms',np.quantile(np.linalg.norm(x,axis=1),[0,.2,.5,.8,1]));
for name,terms in models.items():
 a=design(x,terms); pred=np.empty_like(y)
 for test in np.array_split(np.random.default_rng(608).permutation(len(x)),5):
  train=np.setdiff1d(np.arange(len(x)),test); pred[test]=a[test]@robust(a[train],y[train]);
 r=np.linalg.norm(pred-y,axis=1); print(name,'OOF',np.quantile(r,[.25,.5,.75,.85,.9,.95,1]),'low/mid/high', [np.median(r[i]) for i in np.array_split(np.argsort(np.linalg.norm(x,axis=1)),3)]); coef=robust(a,y); print('coef',coef if len(terms)<=15 else np.c_[np.linalg.norm(coef,axis=1)])
for j in range(4):
 mask=b['bank_id']==j; q=b['ref_signal'][mask,None]; t=b['raw_readout'][mask,None]; a=np.c_[q**0,q,q**2,q**3]; cc=robust(a,t); print('bank',j,cc.ravel(),np.sort(np.abs(t-a@cc).ravel()))
terms=models['mixed 3']; a=design(x,terms); co=robust(a,y); print('active monomials');
for j in range(4): print(j,[(t,round(v,6)) for t,v in zip(terms,co[:,j]) if abs(v)>1e-3]);
