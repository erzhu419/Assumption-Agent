import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; sb=c['single_bank_sweeps']; np.set_printoptions(precision=6,suppress=True)
idx=np.arange(len(x)); obvious=np.max(np.abs(y),axis=1)>8
print('gross rows',idx[obvious].tolist(), 'largest ordinary-looking rows',sorted(np.max(np.abs(y[~obvious]),axis=1))[-6:])
models={d:[()]+[(i,) for i in range(4)] for d in ('affine','diag_cubic','separable_cubic','full_cubic')}
models['separable_cubic'] += [(i,)*j for j in (2,3) for i in range(4)]
models['full_cubic'] += [p for j in (2,3) for p in combinations_with_replacement(range(4),j)]
for name,terms in models.items():
 e=np.empty_like(y); pred=np.empty_like(y); cs=[]
 for j in range(4):
  t=terms+([(j,j),(j,j,j)] if name=='diag_cubic' else [])
  a=np.column_stack([np.prod(x[:,p],axis=1) for p in t]); b=np.linalg.lstsq(a[~obvious],y[~obvious,j],rcond=None)[0]; cs.append(b); pred[:,j]=a@b
  for fold in np.array_split(np.random.default_rng(21).permutation(idx),5):
   tr=(~obvious)&(~np.isin(idx,fold)); e[fold,j]=a[fold]@np.linalg.lstsq(a[tr],y[tr,j],rcond=None)[0]-y[fold,j]
 print(name,'ordinary OOF RMSE',np.sqrt(np.mean(e[~obvious]**2,axis=0)),'max',np.max(abs(e[~obvious])),'axis/mixed max',[np.max(abs(e[(~obvious)&(mask)])) for mask in (idx<16,idx>=16)])
 if name=='diag_cubic':
  print('coefficients (columns=output, rows=offset, 4 linear, own square, own cube):\n',np.array(cs).T)
  k=np.array(cs); bank=sb['bank_id'].astype(int); t=sb['ref_signal']; fit=k[bank,0]+k[bank,bank+1]*t+k[bank,5]*t*t+k[bank,6]*t*t*t; print('independent bank residuals:\n',(sb['raw_readout']-fit).reshape(4,-1)); print('ordinary largest residual rows',np.argsort(np.max(abs(pred-y),axis=1))[-12:].tolist())
if __name__=='__main__':
 print('sensor dimensions',s['nx'],s['ny'],s['source_modes'].shape,len(s['mu_shifts']),s['sensor_ij'].shape)
