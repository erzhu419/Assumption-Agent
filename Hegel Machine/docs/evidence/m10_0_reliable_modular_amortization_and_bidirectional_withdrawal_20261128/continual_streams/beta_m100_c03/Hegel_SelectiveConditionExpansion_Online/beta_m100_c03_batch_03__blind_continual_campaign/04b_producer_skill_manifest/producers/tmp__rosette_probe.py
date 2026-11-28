import itertools as it,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; good=np.max(abs(y),axis=1)<8; axis=np.count_nonzero(x,axis=1)==1
print('gross rows',np.flatnonzero(~good)); print('bank ref/raw first 10',np.c_[c['single_bank_sweeps']['bank_id'],c['single_bank_sweeps']['ref_signal'],c['single_bank_sweeps']['raw_readout']][:10]); print('observation head',s['y_obs_raw'].reshape(-1,4)[:5])
terms=[()] + [t for k in range(1,6) for t in it.combinations_with_replacement(range(4),k)]; A=np.column_stack([np.prod(x[:,t],axis=1) for t in terms]); rng=np.random.default_rng(991); splits=np.array_split(rng.permutation(np.flatnonzero(good)),5)
for name,idx in [('affine',list(range(5))),('diagonal cubic',list(range(5))+[j for j,t in enumerate(terms) if len(t)>1 and len(t)<=3 and len(set(t))==1]),('full quad',list(range(15))),('cubic pure + quad',list(range(15))+[j for j,t in enumerate(terms) if len(t)==3 and len(set(t))==1]),('full cubic',list(range(35))),('full quartic',list(range(70)))]:
 R=np.zeros_like(y)
 for test in splits:
  tr=good.copy(); tr[test]=0; F=A[:,idx]; R[test]=F[test]@np.linalg.lstsq(F[tr],y[tr],rcond=None)[0]-y[test]
 coef=np.linalg.lstsq(A[good][:,idx],y[good],rcond=None)[0]; print(name,'OOF channel rmse',np.round(np.sqrt(np.mean(R[good]**2,0)),6),'max axis/mixed',np.max(abs(R[axis&good])),np.max(abs(R[~axis&good]))); 
 if name=='full cubic':
  for j,v in zip(idx,coef): print(terms[j],np.round(v,6))
