import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; sb=c['single_bank_sweeps']; j=sb['bank_id'].astype(int)
t=[()]+[p for d in (1,2,3) for p in combinations_with_replacement(range(4),d)]
def basis(x): return np.column_stack([np.prod(x[:,p],axis=1) for p in t])
a=basis(x); mag=np.abs(y).max(1); good=mag<8
b=np.linalg.lstsq(a[good],y[good],rcond=None)[0]
print('terms with any coefficient > 0.001:'); print('\n'.join(f'{p}: {r}' for p,r in zip(t,b.round(7)) if np.max(abs(r))>0.001))
print('max negligible coefficient',np.max(abs(b)[abs(b)<0.001])); print('nonnegligible minimum',np.min(abs(b)[abs(b)>0.001]))
xb=np.zeros((len(j),4)); xb[np.arange(len(j)),j]=sb['ref_signal']; ab=basis(xb)
err=np.empty_like(y); errs=[]; supports=[]
for fold in np.array_split(np.random.default_rng(21).permutation(len(x)),5):
 tr=good.copy(); tr[fold]=False
 cv=np.linalg.lstsq(a[tr],y[tr],rcond=None)[0]; bc=np.zeros_like(cv); supports.append(abs(cv)>0.001)
 for k in range(4):
  m=abs(cv[:,k])>0.001; bc[m,k]=np.linalg.lstsq(a[tr][:,m],y[tr,k],rcond=None)[0]
 err[fold]=a[fold]@bc-y[fold]; errs.append(((ab@bc)[np.arange(len(j)),j]-sb['raw_readout']).reshape(4,-1))
print('supports agree:', all(np.array_equal(m,supports[0]) for m in supports)); print('ordinary row OOF RMSE and max:',np.sqrt(np.mean(err[good]**2)),abs(err[good]).max()); print('held-out bank abs residuals, worst across folds:',np.max(np.abs(errs),axis=0)); print('gross residuals:',err[~good]); print('robin',s['robin_x'],s['robin_y'],'shifts',s['mu_shifts']); print('public assets:',len(pickle.load(open('/holdout_task/public_batch.pkl','rb'))))
