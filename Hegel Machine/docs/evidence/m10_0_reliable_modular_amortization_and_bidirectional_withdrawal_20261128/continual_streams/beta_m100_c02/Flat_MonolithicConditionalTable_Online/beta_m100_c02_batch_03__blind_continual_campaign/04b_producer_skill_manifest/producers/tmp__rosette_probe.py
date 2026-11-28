import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['sensor_calibration_sweeps']; X=a['ref_rosette']; Y=a['raw_rosette']; B=a['single_bank_sweeps']; eye=np.eye(4); print('ref range',np.min(X,0),np.max(X,0)); print('current telemetry',np.round(s['aux_manifold_telemetry'],2));
for i in range(4):
 ix=B['bank_id']==i; x=B['ref_signal'][ix]; y=B['raw_readout'][ix]; z=np.stack([x**0,x,x**2,x**3],axis=1); good=abs(y)<5; p=np.linalg.lstsq(z[good],y[good],rcond=None)[0]; print('bank',i,'coeff',p.round(6),'resid',(y-z@p).round(5))
 for name,terms in [('affine',[(j,) for j in range(4)]),('cyclic',[(j,) for j in range(4)]+[(i,i,i),(i,(i+1)%4)]),('quadratics+cubes',list(combinations_with_replacement(range(4),2))+[(j,) for j in range(4)]+[(j,j,j) for j in range(4)]),('all cubic',sum([list(combinations_with_replacement(range(4),k)) for k in (1,2,3)],[]))]:
  F=np.column_stack([np.ones(len(X))]+[np.prod(X[:,t],axis=1) for t in terms]); e=[]; pred=np.empty(len(Y)); rng=np.random.default_rng(743)
  for test in np.array_split(rng.permutation(len(Y)),5):
   train=np.setdiff1d(np.arange(len(Y)),test); train=train[abs(Y[train,i])<8]; v=np.linalg.lstsq(F[train],Y[train,i],rcond=None)[0]; pred[test]=F[test]@v
  r=Y[:,i]-pred; good=abs(Y[:,i])<8; v=np.linalg.lstsq(F[good],Y[good,i],rcond=None)[0]; print(name,'cv',np.quantile(abs(r[good]),[.5,.95,1]).round(6),'terms',[(t,round(q,4)) for t,q in zip([()]+terms,v) if abs(q)>.006])
